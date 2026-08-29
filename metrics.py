def levenshtein_distance(s1, s2):
    """
    Computes Levenshtein distance between two sequences (strings or lists)
    using dynamic programming without external C++ libraries.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def compute_cer(predictions, ground_truths):
    """
    Computes Character Error Rate (CER) = Total Character Edits / Total Ground Truth Characters
    """
    total_edits = 0
    total_len = 0
    for pred, gt in zip(predictions, ground_truths):
        total_edits += levenshtein_distance(pred, gt)
        total_len += len(gt)
    return (total_edits / max(1, total_len)) * 100.0

def compute_wer(predictions, ground_truths):
    """
    Computes Word Error Rate (WER) = Total Word Edits / Total Ground Truth Words
    """
    total_edits = 0
    total_len = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.strip().split()
        gt_words = gt.strip().split()
        total_edits += levenshtein_distance(pred_words, gt_words)
        total_len += len(gt_words)
    return (total_edits / max(1, total_len)) * 100.0