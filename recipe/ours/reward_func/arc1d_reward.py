import ast
import operator
import random
import re


def extract_solution(solution_str):
    """Extract the answer_array from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def extract_numbers(answer_str):
    """
    从 <answer> 标签中间提取出的字符串中鲁棒地提取数字列表。
    支持形如 '[1, 2, 3]', '(1, 2, 3)', '1 2 3', '1,2,3' 等格式。
    """
    # 匹配整数或浮点数，包括负号
    return [int(x) for x in re.findall(r'-?\d+', answer_str)]



def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., data_source='', extra_info=None):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing answer array and size number
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    gtanswer = ground_truth['answer']
    size = ground_truth['size']
    
    answer_array = extract_solution(solution_str=solution_str)
    do_print = 0#random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Answer: {gtanswer} | Numbers: {size}")
        print(f"Extracted answer_array: {answer_array}")
        print(f"Solution string: {solution_str}")

    if answer_array is None:
        if do_print:
            print(f"No answer_array found")
        return 0
    
    try:
        processed_answer_array = extract_numbers(answer_array)
        processed_gt_answer_array = list(gtanswer)
        if processed_answer_array == processed_gt_answer_array:
            if do_print:
                print(f"Correct answer_array: {answer_array} matches ground truth: {gtanswer}")
            return score
        if len(processed_answer_array) == size:
            if do_print:
                print(f"Answer array size match")
            return format_score
        return 0
    except:
        if do_print:
            print(f"Error evaluating answer_array")
        return 0