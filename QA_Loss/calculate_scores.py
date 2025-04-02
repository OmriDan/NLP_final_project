import json

def calculate_sums(lst):
    min_index = lst.index(min(lst))
    sum_before_min = sum(lst[:min_index])
    return sum_before_min

def add_singular_metrics(filename):
    dataset = json.load(open(filename, 'r'))

    for data in dataset:
        min_loss = min(data['qa_loss_sequence'])
        data['MinLossIdx'] = 1.0 * data['qa_loss_sequence'].index(min_loss)
        data['LossRange'] = max(data['qa_loss_sequence']) - min_loss
        data['FullLoss'] = data['qa_loss_sequence'][-1]
        data['MinLossAUC'] = calculate_sums([loss - min_loss for loss in data['qa_loss_sequence']])

    return dataset

def add_additional_metrics_fairytaleqa(filename):
    dataset = add_singular_metrics(filename)

    if 'flan_t5_0.8b' in filename:
        amateur_f = 'context_necessity_flan_t5_0.8b.json'
        expert_f = 'context_necessity_flan_t5_3b.json'
    elif 'flan_t5_3b' in filename:
        amateur_f = 'context_necessity_flan_t5_3b.json'
        expert_f = 'context_necessity_flan_t5_11b.json'
    elif 'flan_t5_11b' in filename:
        amateur_f = 'context_necessity_flan_t5_3b.json'
        expert_f = 'context_necessity_flan_t5_11b.json'
    elif 'qwen_0.5b' in filename:
        amateur_f = 'context_necessity_qwen_0.5b.json'
        expert_f = 'context_necessity_qwen_1.8b.json'
    elif 'qwen_1.8b' in filename:
        amateur_f = 'context_necessity_qwen_1.8b.json'
        expert_f = 'context_necessity_qwen_4b.json'
    elif 'qwen_4b' in filename:
        amateur_f = 'context_necessity_qwen_4b.json'
        expert_f = 'context_necessity_qwen_7b.json'
    elif 'qwen_7b' in filename:
        amateur_f = 'context_necessity_qwen_7b.json'
        expert_f = 'context_necessity_qwen_14b.json'
    elif 'qwen_14b' in filename:
        amateur_f = 'context_necessity_qwen_7b.json'
        expert_f = 'context_necessity_qwen_14b.json'

    dataset_amateur = add_singular_metrics(amateur_f)
    dataset_expert = add_singular_metrics(expert_f)
    values = [d_s['FullLoss'] - d_b['FullLoss'] for d_s, d_b in zip(dataset_amateur, dataset_expert)]
    for data, v in zip(dataset, values):
        data['ExpertiseGap'] = v
    
    return dataset