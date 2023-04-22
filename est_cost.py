import ast
import json
import numpy as np
from transformers import GPT2Tokenizer


def compute_avg_gen(generated: list) -> float:
    return np.mean([len(tokenizer.encode(str(x))) for x in generated])


def est_cost(total_token: int) -> float:
    return total_token / 1000 * 0.02


def main():
    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    counter = 0

    print('-------------------------------------------------------------------------')
    # ANLI
    anli_generated = open('./datasets/ANLI/result/text_davinci003_gold_3800_None.txt', 'r').readlines()
    print(f'ANLI: There are {len(anli_generated)} generated samples')
    anli_avg = compute_avg_gen(anli_generated)
    anli_cost = est_cost((anli_avg + 1000) * 1000)
    counter += 1

    # HellaSWAG
    hellaswag_generated = open('./datasets/HellaSWAG/pred_davinci_text_4000.txt', 'r').readlines()
    print(f'HellaSWAG: There are {len(hellaswag_generated)} generated samples')
    hellaseag_avg = compute_avg_gen(hellaswag_generated)
    hellaswag_cost = est_cost((hellaseag_avg + 1000) * 1000)
    counter += 1

    # HotpotQA
    hotpotqa_dict = json.load(open('./datasets/HotpotQA/result/text_davinci003_pred_3800_None.json', 'r'))['answer']
    hotpotqa_generated = list(hotpotqa_dict.values())
    print(f'HotpotQA: There are {len(hotpotqa_generated)} generated samples')
    hotpotqa_avg = compute_avg_gen(hotpotqa_generated)
    hotpotqa_cost = est_cost((hotpotqa_avg + 1000) * 1000)
    counter += 1

    # OpenPI
    openpi_list = []
    with open('./datasets/OpenPI-v2/result/text_davinci_pred_3500_None.jsonl', 'r') as f:
        for line in list(f):
            openpi_list.append(json.loads(line))
    print(f'OpenPI: There are {len(openpi_list)} generated samples')
    openpi_generated = [item['answers'] for item in openpi_list]
    openpi_generated = [' '.join(item) for item in openpi_generated]
    openpi_avg = compute_avg_gen(openpi_generated)
    openpi_cost = est_cost((openpi_avg + 1000) * 1000)
    counter += 1

    # WinoGrande
    wg_generated = open('./datasets/Winogrande/result/text_davinci003_pred_3800_None.txt', 'r').readlines()
    print(f'WinoGrande: There are {len(wg_generated)} generated samples')
    wg_avg = compute_avg_gen(wg_generated)
    wg_cost = est_cost((wg_avg + 1000) * 1000)
    counter += 1

    # CNN/DailyMail
    cd_generated = open('./datasets/cnn_dailymail/pred-davinci-4000.txt', 'r').readlines()
    print(f'CNN/DailyMail: There are {len(cd_generated)} generated samples')
    cd_avg = compute_avg_gen(cd_generated)
    cd_cost = est_cost((cd_avg + 1000) * 1000)
    counter += 1

    # imdb
    imdb_generated = open('./datasets/imdb/pred_davinci_text.txt', 'r').readlines()
    print(f'IMDB: There are {len(imdb_generated)} generated samples')
    imdb_avg = compute_avg_gen(imdb_generated)
    imdb_cost = est_cost((imdb_avg + 1000) * 1000)
    counter += 1

    # squad
    squad_generated = open('./datasets/squad/pred-davinci-4000.txt', 'r').readlines()
    print(f'SQuAD: There are {len(squad_generated)} generated samples')
    squad_avg = compute_avg_gen(squad_generated)
    squad_cost = est_cost((squad_avg + 1000) * 1000)
    counter += 1
    
    # wikihow gs
    wkgs_generated = open('./datasets/wikihow_goal_step/pred_davinci003_text_4000.txt', 'r').readlines()
    print(f'WikiHow GS: There are {len(wkgs_generated)} generated samples')
    wkgs_avg = compute_avg_gen(wkgs_generated)
    wkgs_cost = est_cost((wkgs_avg + 1000) * 1000)
    counter += 1

    # wikihow tmp
    wktp_generated = open('./datasets/wikihow_temporal/pred_davinci003_text_4000.txt', 'r').readlines()
    print(f'WikiHow Temporal: There are {len(wktp_generated)} generated samples')
    wktp_avg = compute_avg_gen(wktp_generated)
    wktp_cost = est_cost((wktp_avg + 1000) * 1000)
    counter += 1
    
    # xsum
    xsum_generated = open('./datasets/xsum/pred-davinci-4000.txt', 'r').readlines()
    print(f'XSum: There are {len(xsum_generated)} generated samples')
    xsum_avg = compute_avg_gen(xsum_generated)
    xsum_cost = est_cost((xsum_avg + 1000) * 1000)
    counter += 1

    # yelp
    yelp_generated = open('./datasets/yelp/pred_davinci003_text_4000.txt', 'r').readlines()  
    print(f'Yelp: There are {len(yelp_generated)} generated samples')
    yelp_avg = compute_avg_gen(yelp_generated)
    yelp_cost = est_cost((yelp_avg + 1000) * 1000)
    counter += 1


    print('-------------------------------------------------------------------------')
    print(f'There are {counter} datasets in total')
    print('-------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------')
    print(f'The cost of running ANLI with 1000 tokens in context is ${4 * anli_cost:.3f}')
    print(f'The cost of running HellaSWAG with 1000 tokens in context is ${4 * hellaswag_cost:.3f}')
    print(f'The cost of running HotpotQA with 1000 tokens in context is ${4 * hotpotqa_cost:.3f}')
    print(f'The cost of running OpenPI with 1000 tokens in context is ${4 * openpi_cost:.3f}')
    print(f'The cost of running WinoGrande with 1000 tokens in context is ${4 * wg_cost:.3f}')
    print(f'The cost of running CNN/DailyMail with 1000 tokens in context is ${4 * cd_cost:.3f}')
    print(f'The cost of running IMDB with 1000 tokens in context is ${4 * imdb_cost:.3f}')
    print(f'The cost of running SQuAD with 1000 tokens in context is ${4 * squad_cost:.3f}')
    print(f'The cost of running WikiHow GS with 1000 tokens in context is ${4 * wkgs_cost:.3f}')
    print(f'The cost of running WikiHow Temporal with 1000 tokens in context is ${4 * wktp_cost:.3f}')
    print(f'The cost of running XSum with 1000 tokens in context is ${4 * xsum_cost:.3f}')
    print(f'The cost of running Yelp with 1000 tokens in context is ${4 * yelp_cost:.3f}')

if __name__ == "__main__":
    main()
