import re


def extract_solution(solution_str, method="strict"):
    """Extract answer from text in format 'xxx: answer'

    Args:
        solution_str: Input text in format 'xxx: answer'
        method: Extraction method, choices are 'strict' and 'flexible'

    Returns:
        Extracted answer string or None if extraction fails
    """
    assert method in ["strict", "flexible"]

    try:
        # Split by ':' and take the last part
        parts = solution_str.split(':')
        if len(parts) > 1:
            # Strip whitespace from the answer
            answer = parts[-1].strip()
            answer = set(answer.split("|"))
            return answer
        return None
    except Exception as e:
        print(e)
        return None



def calculate_f1(answer: set, gt: set) -> float:
    """Calculate F1 score given prediction and ground truth sets.

    Args:
        answer (set): Prediction set
        gt (set): Ground truth set

    Returns:
        float: F1 score between 0 and 1
    """
    if len(answer) == 0 and len(gt) == 0:
        return 1.0
    if len(answer) == 0 or len(gt) == 0:
        return 0.0

    intersection = len(answer & gt)
    precision = intersection / len(answer)
    recall = intersection / len(gt)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1



def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, **kwargs):
    # extract to list
    solution_list = solution_str.split(";")
    gt_list = ground_truth.split(";")

    solution_list = [i.strip() for i in solution_list]
    gt_list = [i.strip() for i in gt_list]

    # print(solution_list)
    # cal f1 score for each field
    mean_f1_scores = 0
    for sol, gt in zip(solution_list, gt_list):
        mean_f1_scores += compute_score_each_field(sol, gt, method, format_score, score)

    mean_f1_scores /= len(solution_list)
    # return mean f1 score
    return mean_f1_scores



# For single field prediction
def compute_score_each_field(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, **kwargs):
    """The scoring function for extracting and comparing answers.

    Args:
        solution_str: the solution text in format 'xxx: answer'
        ground_truth: the ground truth answer
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the correct format but wrong answer
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    gt = extract_solution(solution_str=ground_truth, method=method)

    if answer is None or len(answer) == 0:
        return 0
    else:
        # overlap logic
        # subset logic
        # if gt.issubset(answer):
        # exact match / f1 score ==>
        # if answer & gt:
        #     return score
        # else:
        #     return format_score
        return calculate_f1(answer, gt)

if __name__ == '__main__':
    # test
    pred = "incident location: pick area ; primary impact: strain or injury by ; principle body part: trunk ; detailed body part: abdomen ; primary object: box/package ; secondary object: standard package ; risk group and category: health-ergonomic ; risk hazard: awkward posture (technique)"
    gt = "incident location: case receive ; primary impact: strain or injury by ; principle body part: ankle ; detailed body part: achilles|ankle, not otherwise specified|inside of ankle|outside of ankle ; primary object: pit ; secondary object: order picker ; risk group and category: health-ergonomic ; risk hazard: awkward posture (technique)"
    print(compute_score(pred, gt))

    pred = 'incident location: onsite parking lot'
    gt = 'incident location: onsite parking lot'
    print(compute_score(pred, gt))