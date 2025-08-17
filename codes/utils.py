import jiwer
import torch
import copy
import math
import nemo.collections.asr as nemo_asr


def fetch_word_ops(REF, HYP, BS):
    lines = jiwer.visualize_alignment(jiwer.process_words(REF, HYP), show_measures=False, skip_correct=False).split('\n')
    word_ops_list = []
    ref_j_list = []
    hyp_j_list = []
    j = 1
    k = 2
    
    for i in range(BS):
        ref_align = lines[j+(5*i)][5:]
        hyp_align = lines[k+(5*i)][5:]
        
        word_ops = []
        
        ref_j = ref_align.split()
        hyp_j = hyp_align.split()
        ref_j_list.append(ref_j)
        hyp_j_list.append(hyp_j)
        
        for i in range(len(ref_j)):
            if ref_j[i] == hyp_j[i]:
                word_ops.append('C')
            elif '*' in ref_j[i]:
                word_ops.append('I')
            elif '*' in hyp_j[i]:
                word_ops.append('D')
            else:
                word_ops.append('S')
        word_ops_list.append(word_ops)
   
    return word_ops_list, ref_j_list, hyp_j_list


def find_pred_ranges(asr_model, with_blank_ids_list, HYP, vocab, uncollapsed_HYP):
    words_list = []
    for i in range(len(HYP)):
        words = HYP[i].split()
        words_list.append(words)
    # print(words_list)
    temp_hyp_list = []

    for j in range(len(with_blank_ids_list)):   
        temp_hyp = []
        for i in range(len(with_blank_ids_list[j])):
            temp_hyp.append((vocab[with_blank_ids_list[j][i]], i))
        temp_hyp_list.append(temp_hyp)   
    
    col_hyp_list = []
    
    for j in range(len(temp_hyp_list)):
        col_hyp = []
        # Iterate over the original list
        for l in range(len(temp_hyp_list[j])):
            # If it's the first element or the current element is different from the previous one
            if l == 0 or temp_hyp_list[j][l][0] != temp_hyp_list[j][l - 1][0]:
                # Append the current element to the result list
                col_hyp.append(temp_hyp_list[j][l])
        col_hyp_list.append(col_hyp)
    # print(col_hyp_list)

    final_hyp_list = []
    for j in range(len(col_hyp_list)):   
        temp_hyp = []
        for i in range(len(col_hyp_list[j])):
            if col_hyp_list[j][i][0] != '@':
                temp_hyp.append(col_hyp_list[j][i])
        final_hyp_list.append(temp_hyp)   
    # print(final_hyp_list)

    word_offset_list = []
    
    for j in range(len(with_blank_ids_list)):
        words = words_list[j]
        # print(words)
        k = 0
        one_word = words[k]
 
        tokens = []
        iter = 0
        start_offset = 0
        word_offset = []
        while k<=len(words)-1:
            tokens.append(final_hyp_list[j][iter][0])
            # print(tokens)
            if one_word == asr_model.decoding.decode_tokens_to_str(tokens):
                # print(tokens)
                ext = True
                temp_toks = list(tokens)
                temp_iter = iter
                while ext==True:
                    temp_iter+=1
                    if temp_iter == len(final_hyp_list[j]):
                        break
                    temp_toks.append(final_hyp_list[j][temp_iter][0])
                    if one_word == asr_model.decoding.decode_tokens_to_str(temp_toks):
                        ext = True
                    else:
                        ext = False
                        iter = temp_iter-1
                end_offset = final_hyp_list[j][iter][1]
                # print(tokens)
                pipe_word = '|'.join(uncollapsed_HYP[j][start_offset:end_offset+1])
                word_offset.append((pipe_word, start_offset, end_offset))
                start_offset = end_offset + 1
                k = k + 1
                if k == len(words):
                    break
                else:
                    one_word = words[k]
                tokens = []
            iter+=1
            if iter == len(final_hyp_list[j]):
                break
        # print(word_offset)
        word_offset_list.append(word_offset)
    return word_offset_list


def ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab, temperature = 1.0):
    # vocab.append('@')
    unkwn = 0
    blank = len(vocab)-1
    encoder_output = asr_model.encoder(audio_signal=processed_signal_list, length=processed_signal_length_list)
    encoded = encoder_output[0]
    
    if asr_model.decoder.is_adapter_available():
        encoded = encoded.transpose(1, 2)  # [B, T, C]
        encoded = asr_model.forward_enabled_adapters(encoded)
        encoded = encoded.transpose(1, 2)  # [B, C, T]
    
    decoder_out = asr_model.decoder.decoder_layers(encoded)
    
    asr_model.decoder.temperature = temperature

    if asr_model.decoder.temperature != 1.0:
        soft = (decoder_out.transpose(1, 2)/asr_model.decoder.temperature).softmax(dim=-1)
    else:
        soft = decoder_out.transpose(1, 2).softmax(dim=-1)

    greedy_predictions = soft.argmax(dim=-1, keepdim=False)
    HYP = []
    uncollapsed_HYP = []
    with_blank_ids_list = []
    for i in range(greedy_predictions.shape[0]):
        with_blank_ids = [label for label in greedy_predictions[i].tolist()]
       
        greedy_predictions1 = torch.unique_consecutive(greedy_predictions[i], dim=-1)
        non_blank_ids = [label for label in greedy_predictions1.tolist() if label!=blank]
        non_blank_ids = [label for label in non_blank_ids if label!=unkwn] #added aug 31 24
        tokens = asr_model.decoding.decode_ids_to_tokens(non_blank_ids)
        HYP.append(asr_model.decoding.decode_tokens_to_str(tokens))
        
        with_blank_hyp = []
        for i in range(len(with_blank_ids)):
            with_blank_hyp.append(vocab[with_blank_ids[i]])
        
        uncollapsed_HYP.append(with_blank_hyp)
        with_blank_ids_list.append(with_blank_ids)

    return encoded, decoder_out, soft, HYP, uncollapsed_HYP, with_blank_ids_list

def test_validate(batch, asr_model, DEVICE, DEV_BS, vocab, temperature = 1.0):
    
    input_signal_list, input_signal_length_list, references = batch
    processed_signal_list, processed_signal_length_list = asr_model.preprocessor(
                    input_signal=input_signal_list.to(DEVICE), length=torch.tensor(input_signal_length_list, dtype=torch.float32).to(DEVICE),
                ) 
    encoded, decoder_out, soft, HYP, uncollapsed_HYP, with_blank_ids_list = ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab, temperature = temperature) 
    for i in range(len(HYP)):
        if HYP[i] == '':
            return 0, references, HYP, [0], [0], [0]
    
    word_ops_list, ref_j_list, hyp_j_list = fetch_word_ops(references, HYP, DEV_BS)
    word_offset_list = find_pred_ranges(asr_model, with_blank_ids_list, HYP, vocab, uncollapsed_HYP)
    word_offset_list_temp = copy.deepcopy(word_offset_list) 
    soft = soft.permute(0, 2, 1)
    
    softmax_scores = softmax_score_gen(soft, word_offset_list)
    
    return references, HYP, word_ops_list, softmax_scores

# word_offset_list = [[('@|@|▁और', 0, 2), ('@|@|@|@|@|@|▁सं|@|@|@|@|क|@|@|ट', 3, 17), ('@|@|▁के', 18, 20), ('@|@|@|@|▁स|@|@|म|@|@|य', 21, 31), ('@|@|@|@|@|@|@|@|@|@|@|@|@|@|@|@|@|▁न|@|@|य|@|ा', 32, 54), ('@|@|@|@|▁र|@|ा|@|स|@|्|्|ता', 55, 67), ('@|@|@|@|▁भी', 68, 72), ('@|@|@|@|@|▁ब|@|@|ना', 73, 81), ('@|@|@|▁द|@|े|@|@|त|ी', 82, 91), ('@|▁है', 92, 93)]]
def softmax_idx_gen(word_offset_list):
    batch_size = len(word_offset_list)
    N = len(word_offset_list[0])
    soft_idx_batchwise = []
    for b in range(batch_size):
        soft_idx = []
        for i in range(N):
            split = word_offset_list[b][i][0].split("|")
            start_idx = word_offset_list[b][i][1]
            end_idx = word_offset_list[b][i][2]
            word_idx = [j for j in range(start_idx, end_idx+1) if split[j-start_idx] != '@']
            soft_idx.append(word_idx)
        soft_idx_batchwise.append(soft_idx)

    return soft_idx_batchwise

def softmax_score_gen(soft, word_offset_list):
    soft_idx_batchwise = softmax_idx_gen(word_offset_list)
    batch_size = len(soft_idx_batchwise)
    N = len(soft_idx_batchwise[0])
    score_batchwise = []
    for b in range(batch_size):
        score = []
        for i in range(N):
            soft_idx = soft_idx_batchwise[b][i]
            W = len(soft_idx)
            temp  = 0
            for j in range(W):
                temp += soft[b,:,soft_idx[j]].max()
            score.append((temp/W).item())
        score_batchwise.append(score)
    return score_batchwise

def mainf(batch, asr_model, DEVICE, TEST_BS, temperature = 1.0):
    # print("hi")
    # asr_model.decoder.vocabulary
    # print(asr_model)
    blank = len(asr_model.decoder.vocabulary)
    # print(blank)
    unkwn = 0
    vocab = copy.deepcopy(asr_model.decoder.vocabulary)
    vocab.append('@') #For blank character
    # print(vocab)
    references, HYP, word_ops_list, softmax_scores = test_validate(batch, asr_model, DEVICE, TEST_BS, vocab, temperature=temperature)
    return references, HYP, word_ops_list, softmax_scores

def word_idx_gen(word_offset_list):
    batch_size = len(word_offset_list)
    N = len(word_offset_list[0])
    soft_idx_batchwise = []
    for b in range(batch_size):
        soft_idx = []
        for i in range(N):
            split = word_offset_list[b][i][0].split("|")
            start_idx = word_offset_list[b][i][1]
            end_idx = word_offset_list[b][i][2]
            word_idx = [j for j in range(start_idx, end_idx+1) if split[j-start_idx] != '@']
            soft_idx.append(word_idx)
        soft_idx_batchwise.append(soft_idx)
    return soft_idx_batchwise

def entropy_score_gen(soft, word_offset_list, vocab_len, temperature = 0.33, aggregation_method = "prod"):
    frame_idx_batchwise = word_idx_gen(word_offset_list)
    batch_size = len(frame_idx_batchwise)
    N = len(frame_idx_batchwise[0])
    score_batchwise = []
    for b in range(batch_size):
        score = []
        for i in range(N):
            soft_idx = frame_idx_batchwise[b][i]
            W = len(soft_idx)
            temp = []
            prod = 1
            mean = 0
            for j in range(W):
                if aggregation_method == "prod":
                    prod = prod * entropy_tsallis_exp(soft[b,:,soft_idx[j]], V = vocab_len, alpha=temperature)
                elif aggregation_method == 'mean':
                    mean = mean + entropy_tsallis_exp(soft[b,:,soft_idx[j]], V = vocab_len, alpha=temperature)
                else:
                    temp.append(entropy_tsallis_exp(soft[b,:,soft_idx[j]], V = vocab_len, alpha=temperature))
            if aggregation_method == 'prod':
                score.append(prod)
            elif aggregation_method == 'mean':
                score.append(mean/W)
            elif aggregation_method == 'min':
                score.append(min(temp))
            elif aggregation_method == 'max':
                score.append(max(temp))
            else:
                raise RuntimeError('Invalid aggregation method! Choose either of - `min`, `max`, `mean` or `prod`!')
        score_batchwise.append(score)
    return score_batchwise

def entropy_tsallis_exp(x, V = 129, alpha=0.33):
    neg_entropy_alpha = V**(1-alpha) - torch.pow(x, alpha).sum().item()
    denom = math.exp((V**(1-alpha)-1)/(1-alpha))-1
    numer = math.exp(neg_entropy_alpha/(1-alpha))-1
    value = numer/denom
    return value

def test_validate_entropy(batch, asr_model, device, DEV_BS, vocab, temperature = 0.33, aggregation_method = "prod"):
    
    input_signal_list, input_signal_length_list, references = batch
    processed_signal_list, processed_signal_length_list = asr_model.preprocessor(
                    input_signal=input_signal_list.to(device), length=torch.tensor(input_signal_length_list, dtype=torch.float32).to(device),
                ) 
    encoded, decoder_out, soft, HYP, uncollapsed_HYP, with_blank_ids_list = ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab) 
    for i in range(len(HYP)):
        if HYP[i] == '':
            return 0, references, HYP, [0], [0], [0]
        
    word_ops_list, ref_j_list, hyp_j_list = fetch_word_ops(references, HYP, DEV_BS)
    word_offset_list = find_pred_ranges(asr_model, with_blank_ids_list, HYP, vocab, uncollapsed_HYP)
    word_offset_list_temp = copy.deepcopy(word_offset_list) 
    soft = soft.permute(0, 2, 1)
    
    entropy_scores = entropy_score_gen(soft, word_offset_list, len(vocab), temperature=temperature, aggregation_method=aggregation_method)
    
    return references, HYP, word_ops_list, entropy_scores

def mainf_entropy(batch, asr_model, DEVICE, TEST_BS, temperature = 0.33, aggregation_method="mean"):
    # print("hi")
    # asr_model.decoder.vocabulary
    # print(asr_model)
    blank = len(asr_model.decoder.vocabulary)
    # print(blank)
    unkwn = 0
    vocab = copy.deepcopy(asr_model.decoder.vocabulary)
    vocab.append('@') #For blank character
    # print(vocab)
    references, HYP, word_ops_list, softmax_scores = test_validate_entropy(batch, asr_model, DEVICE, TEST_BS, vocab, temperature=temperature, aggregation_method = aggregation_method)
    return references, HYP, word_ops_list, softmax_scores
