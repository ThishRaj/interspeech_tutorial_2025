import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Sample data (replace with your actual score lists)
# correct = [0.9, 0.85, 0.95, 0.8]
# insert = [0.3, 0.25, 0.4]
# subs_c_less_s = [0.45, 0.5, 0.35]
# subs_c_equal_s = [0.55, 0.6]
# subs_c_great_s = [0.2, 0.15, 0.3]
def plot_histograms(y_score_correct, y_score_incorrect, x_label_name = "Softmax Score"):
    fig = plt.figure(figsize=(16,8))


    plt.rcParams.update({'font.size': 40})
    font = {   'weight':'bold',
            'size':40}
    plt.rc('font',**font)
    plt.rc('lines', linewidth = 4)
    plt.rc('xtick.major', size = 5, pad = 7)
    plt.rc('xtick', labelsize = 25)
    plt.rc('ytick.major', size = 5, pad = 7)
    plt.rc('ytick', labelsize = 25)
    ax = fig.gca()

    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed', linewidth=2)
    # Combine into incorrect
    # incorrect = insert + subs_c_less_s + subs_c_equal_s + subs_c_great_s

    # Plotting
    plt.hist(y_score_correct, bins=10, range=(0, 1), density=True, alpha=1, color='green', label='Correct Words')
    # plt.hist(incorrect, bins=20, range=(0, 1), density=True, alpha=0.5, color='red', label='Incorrect')

    # Labels and legend
    plt.xlabel(x_label_name)
    plt.ylabel('Density')
    # plt.title('Distribution of TruCLeS Scores for correct words for CTC models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    # plt.hist(correct, bins=20, range=(0, 1), density=True, alpha=0.5, color='green', label='Correct')
    plt.hist(y_score_incorrect, bins=10, range=(0, 1), density=True, alpha=1, color='red', label='Incorrect Words')

    # Labels and legend
    plt.xlabel(x_label_name)
    plt.ylabel('Density')
    # plt.title('Distribution of TruCLeS Scores for wrong words for CTC models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calc_bins(preds, confs):
  # Assign each prediction to a bin
    preds = np.array(preds)
    confs = np.array(confs)
    num_bins = 11
    bins = np.linspace(start = 0.0, stop= 1.0, num = num_bins)
    binned = np.digitize(preds, bins)
    # np.digitize - Returns the indices of the bins to which each value in input array belongs
    
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    
    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (confs[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, confs):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, confs)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE

def draw_reliability_graph(preds, confs):
#     ECE, MCE = get_metrics(preds, confs)
#     bins, _, bin_accs, _, _ = calc_bins(preds, confs)

    ECE, MCE = get_metrics(preds, confs)
    bins, _, bin_accs, _, _ = calc_bins(preds, confs)
    font = {'weight':'bold',
           'size':40}
    plt.rc('font',**font)
    plt.rc('lines', linewidth = 4)
    plt.rc('xtick.major', size = 5, pad = 7)
    plt.rc('xtick', labelsize = 30)
    plt.rc('ytick.major', size = 5, pad = 7)
    plt.rc('ytick', labelsize = 30)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    #ax.set_xlim(-0.05, 1.05)
    ax.set_xlim(0, 1)   #CHANGED
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed', linewidth=2)

    # Error bars
    plt.bar(bins, bins,  width=-0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\', align='edge')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=-0.1, alpha=0.6, edgecolor='black', color='b', align='edge')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    font = {'weight':'bold',
           'size':25}
    plt.rc('font',**font)
    ECE_patch = mpatches.Patch(color='blue', label='Outputs', alpha=0.6)
    MCE_patch = mpatches.Patch(color='red', label='Gaps', alpha=0.3, hatch='\\')
    plt.legend(handles=[ECE_patch, MCE_patch])

    plt.show()

    #plt.savefig('calibrated_network.png', bbox_inches='tight')
    return ECE, MCE