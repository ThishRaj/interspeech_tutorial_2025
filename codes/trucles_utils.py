import torch
import torch.nn.functional as F
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from torch.autograd import Variable

import pandas as pd
import os

import numpy as np
import jiwer
import copy

def save_checkpoint_confid(model, optimizer, valid_loss, epoch, checkpoint_path):
    ''' Save model checkpoint '''
    torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'encoder_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

def get_jaccard(ref, hyp):
    list1 = list(ref)
    list2 = list(hyp)
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def levenshtein_distance(str1, str2):
    # Initialize a matrix to store the distances
    distance_matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Initialize the first row and column
    for i in range(len(str1) + 1):
        distance_matrix[i][0] = i
    for j in range(len(str2) + 1):
        distance_matrix[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1,  # deletion
                                         distance_matrix[i][j - 1] + 1,  # insertion
                                         distance_matrix[i - 1][j - 1] + cost)  # substitution

    # Return the bottom-right cell of the matrix
    return distance_matrix[len(str1)][len(str2)]

def fetch_word_ops(REF, HYP, TRAIN_BS):
    lines = jiwer.visualize_alignment(jiwer.process_words(REF, HYP), show_measures=False, skip_correct=False).split('\n')
    word_ops_list = []
    ref_j_list = []
    hyp_j_list = []
    j = 1
    k = 2
    
    for i in range(TRAIN_BS):
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
    # print(with_blank_ids_list)
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
    # print(temp_hyp_list)

    # New list to store the result
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
        # try:
        #     one_word = words[k]
        # except Exception as e:
        #     print(words)
        #     print(HYP)
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

def fetch_word_ops_single(REF, HYP):
    lines = jiwer.visualize_alignment(jiwer.process_words(REF, HYP), show_measures=False, skip_correct=False).split('\n')
    # print(REF)
    # print(HYP)
    # print(lines)
    ref_align = lines[1][5:]
    hyp_align = lines[2][   5:]
    
    word_ops = []
    
    ref_j = ref_align.split()
    hyp_j = hyp_align.split()
    
    for i in range(len(ref_j)):
        if ref_j[i] == hyp_j[i]:
            word_ops.append('C')
        elif '*' in ref_j[i]:
            word_ops.append('I')
        elif '*' in hyp_j[i]:
            word_ops.append('D')
        else:
            word_ops.append('S')
    # print(ref_j)
    # print(hyp_j)
    # print(word_ops)

    return ref_j, hyp_j, word_ops

def actual_score_gen(references, HYP, uncollapsed_HYP, word_ops_list, ref_j_list, hyp_j_list, word_offset_list_temp, soft, vocab, sp):
      
    del_positions_list = []
    for i in range(len(word_ops_list)):
        del_positions = [index for index, char in enumerate(word_ops_list[i]) if char == 'D']
        del_positions_list.append(del_positions)
          
    tuple_to_insert = ('$', '$', '$')
    # Insert the tuple at each specified position
    for i in range(len(del_positions_list)):
        for pos in del_positions_list[i]:
            word_offset_list_temp[i].insert(pos, tuple_to_insert)
        
    word_score_list = []
    # print(soft.shape)
    for m in range(len(hyp_j_list)):
        # print(soft[m].shape)
        soft_temp = soft[m]
        soft_temp = soft_temp.unsqueeze(0)
        word_score = []
    
        for i in range(len(hyp_j_list[m])):
            if word_ops_list[m][i] == 'D':
                # print(ref_j[i], ',', hyp_j[i], ',', word_ops[i])
                continue
            if word_ops_list[m][i] == 'I':
                score = 0
                # print(ref_j[i], ',', hyp_j[i], ',', word_ops[i], ',', score)
                word_score.append(score)
        
            if word_ops_list[m][i] == 'S':
                score = 0
                weight = 0
                start_offset = word_offset_list_temp[m][i][1]
                end_offset = word_offset_list_temp[m][i][2]
                ref_toks = sp.encode_as_pieces(ref_j_list[m][i])
                # print(ref_toks)
                word = word_offset_list_temp[m][i][0]
                token_list = word.split('|')
        
                # Create a list with numbers in the specified range
                number_list = [num for num in range(start_offset, end_offset + 1)]
                
                indices_list = []
                for j in range(len(number_list)):
                    if token_list[j]=='@':
                        continue
                    else:
                        indices_list.append(number_list[j])
        
                char_to_remove = '@'
                hyp_toks = [string for string in token_list if string != char_to_remove]
                # print(hyp_toks)
                
                merged_ref = " ".join(ref_toks)
                merged_hyp = " ".join(hyp_toks)
        
                ref_subs_j, hyp_subs_j, word_subs_ops = fetch_word_ops_single(merged_ref, merged_hyp)   
                # print(ref_subs_j)
                # print(hyp_subs_j)
                # print(word_subs_ops)
        
                # subs_score = []
                del_count = 0
                for k in range(len(hyp_subs_j)):
                    if word_subs_ops[k]=='D':
                        del_count+=1
                        continue
                        
                    if word_subs_ops[k]=='I':
                        weight+=1
                        score+=0
                        # subs_score.append(0)
                        # print(ref_subs_j[k], ',', hyp_subs_j[k], ',', word_subs_ops[k], 0)
        
                    if word_subs_ops[k]=='C':
                        weight+=1
                        score+=soft_temp[:,indices_list[k-del_count]].max(dim=-1, keepdim=False).values.item()
                        # subs_score.append(soft[:,indices_list[k-del_count]].max(dim=-1, keepdim=False).values.item())
                        # print(ref_subs_j[k], ',', hyp_subs_j[k], ',', word_subs_ops[k], soft[:,indices_list[k-del_count]].max(dim=-1, keepdim=False).values.item())
                        
                    if word_subs_ops[k]=='S':
                        true_token_index = vocab.index(ref_subs_j[k])
                        tcp = soft_temp[:,indices_list[k-del_count]].view(-1)[true_token_index].item()
                        mcp = soft_temp[:,indices_list[k-del_count]].max(dim=-1, keepdim=False).values.item()
                        weight+=(mcp-tcp)
                        score+=(tcp*(mcp-tcp))
                        # subs_score.append(tcp*(mcp-tcp))
                        # print(ref_subs_j[k], ',', hyp_subs_j[k], ',', word_subs_ops[k], tcp*(mcp-tcp))
                # print(subs_score)
                # Calculate Levenshtein distance
                lev_distance = levenshtein_distance(''.join(ref_toks), ''.join(hyp_toks))
                
                # Calculate normalized Levenshtein distance
                max_len = max(len(''.join(ref_toks)), len(''.join(hyp_toks)))
                normalized_lev_distance = lev_distance / max_len
                
                # score = (score/weight)*get_jaccard(''.join(ref_toks), ''.join(hyp_toks))   
                score = (score/weight)*(1-normalized_lev_distance)
                word_score.append(score)
                # print(ref_j[i], ',', hyp_j[i], ',', word_ops[i], ',', ref_toks, ',', score)
        
            if word_ops_list[m][i] == 'C':
                score = 0
        
                start_offset = word_offset_list_temp[m][i][1]
                weight = 0
                token_list = word_offset_list_temp[m][i][0].split('|')
                for l in range(len(token_list)):
                    if token_list[l] == '@':
                        continue
                    else:
                        weight+=1 
                        score+=soft_temp[:,l+start_offset].max(dim=-1, keepdim=False).values.item()
                score = score/weight
                word_score.append(score)
                # print(ref_j[i], ',', hyp_j[i], ',', word_ops[i], ',', score)
        word_score_list.append(word_score)
    return word_score_list

def ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab):
    # vocab.append('@')

    blank = len(vocab)-1
    unkwn = 0
    encoder_output = asr_model.encoder(audio_signal=processed_signal_list, length=processed_signal_length_list)
    encoded = encoder_output[0]
    
    if asr_model.decoder.is_adapter_available():
        encoded = encoded.transpose(1, 2)  # [B, T, C]
        encoded = asr_model.forward_enabled_adapters(encoded)
        encoded = encoded.transpose(1, 2)  # [B, C, T]
    
    decoder_out = asr_model.decoder.decoder_layers(encoded)
   
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

def load_checkpoint_confid(model, checkpoint_path, device):
    ''' Load model checkpoint '''
    if not os.path.exists(checkpoint_path):
        raise 'Checkpoint does not exist'
    checkpoint = torch.load(checkpoint_path, map_location=device)
   # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['encoder_state_dict'])
    return checkpoint["epoch"], checkpoint['valid_loss']


def test_validate_trucles(confid_model, batch, asr_model, device, DEV_BS, vocab, sp):
    confid_model.eval()
    
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
    actual_score = actual_score_gen(references, HYP, uncollapsed_HYP, word_ops_list, ref_j_list, hyp_j_list, word_offset_list_temp, soft, vocab, sp)
    soft = soft.permute(0, 2, 1)
    # print("WHOLE TENSOR")
    # print(encoded.shape)
    # print(decoder_out.shape)
    # print(soft.shape)
    
    confid_tensor = []
    final_loss = 0
    input_stack = []
    
    for i in range(len(word_offset_list)):
        for j in range(len(word_offset_list[i])):
            start_frame = word_offset_list[i][j][1]
            
            end_frame = word_offset_list[i][j][2]
            
            indices = [*range(start_frame, end_frame+1, 1)]
            # print(indices)
            encoded_input = encoded[i,:,indices]
            decoder_input = decoder_out[i,:,indices]
            soft_input = soft[i,:,indices]
            # print("EXTRACTED TENSOR")
            # print(encoded_input.shape)
            # print(decoder_input.shape)
            # print(soft_input.shape)
            encoded_input = torch.mean(encoded_input, 1, keepdim=False) #If false, shape is [256]. If True, shape is [1,256]
            decoder_input = torch.mean(decoder_input, 1, keepdim=False)
            soft_input = torch.mean(soft_input, 1, keepdim=False)
            # print("MEAN TENSOR")
            # print(encoded_input.shape)
            # print(decoder_input.shape)
            # print(soft_input.shape)
            confid_input = [encoded_input, decoder_input, soft_input]
            confid_tensor = torch.cat(confid_input, dim=-1)
            # print("CONFID TENSOR")
            # print(confid_tensor.shape)
            input_stack.append(confid_tensor)
    
    input_stack = torch.stack(input_stack)
    score = confid_model(input_stack)

    actual_score_for_target = [] 
    for sublist in actual_score:
        for item in sublist:
            actual_score_for_target.append(item)
    target_score = Variable(torch.FloatTensor(actual_score_for_target), requires_grad = True).to(device)

    

    return references, HYP, word_ops_list, actual_score, score.flatten().tolist()