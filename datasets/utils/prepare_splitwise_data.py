import click
import random
import pandas as pd
from utils import (dataset_dump, gen_random_tokens, get_list_of_delays,
                         get_norm_dist_tokens)


@click.command()
@click.option("--num-requests",
              required=True,
              type=int,
              help='Number of requests to be generated')
@click.option('--trace-path',
              required=True,
              type=str,
              help='Path for the Splitwise trace file')
@click.option("--mode",
              type=click.Choice(['token', 'text', 'length']),
              default='token',
              help='Mode of the dataset generation',
              )

@click.pass_obj
def splitwise(root_args, **kwargs):
    """Prepare dataset by generating random tokens."""
    input_ids = []
    input_lens = []
    output_lens = []

    # input_lens = get_norm_dist_tokens(kwargs['input_mean'],
    #                                   kwargs['input_stdev'],
    #                                   kwargs['num_requests'],
    #                                   root_args.random_seed)

    # parse the trace file into a pandas dataframe
    df = pd.read_csv(kwargs['trace_path'], sep=',')
    # get the ContextTokens from the trace file as a list, the format is TIMESTAMP,ContextTokens,GeneratedTokens
    input_lens = df['ContextTokens'].tolist()
    output_lens = df['GeneratedTokens'].tolist()
    # we filter out the requests that are longer than the total-len
    remove_indices = []
    for i in range(len(input_lens)):
        if input_lens[i] + output_lens[i] > root_args.total_len:
            remove_indices.append(i)
        elif input_lens[i] > 3072 or output_lens[i] > 1024:
            remove_indices.append(i)
    # remove indices from the input_lens and output_lens
    for index in sorted(remove_indices, reverse=True):
        del input_lens[index]
        del output_lens[index]
    # now use root_args.random_seed to pick num_requests of ids from the input_ids
    if len(input_lens) < kwargs['num_requests']:
        raise ValueError(
            f"Number of requests required is more than the number of requests in the trace file. Number of requests in the trace file: {len(input_lens)}"
        )
    random.seed(root_args.random_seed)
    ids = list(range(len(input_lens)))
    selected_ids = random.sample(ids, kwargs['num_requests'])
    input_lens = [input_lens[i] for i in selected_ids]
    output_lens = [output_lens[i] for i in selected_ids]
    
    # for i in range(len(input_lens)):
    #     print(f"input_lens: {input_lens[i]}, output_lens: {output_lens[i]}")
    num_reqs = len(input_lens)

    delays = get_list_of_delays(root_args.time_delay_dist,
                                root_args.mean_time_bet_reqs, num_reqs,
                                root_args.random_seed)

    if kwargs['mode'] == 'token':
        input_ids = gen_random_tokens(input_lens, root_args.tokenizer,
                                    root_args.random_seed)

        dataset_dump(
            input_ids, output_lens, delays, {
                "workload_type": "splitwise",
                "num_requests": kwargs['num_requests'],
                "delay_distr": root_args.time_delay_dist,
                "request_rate": root_args.request_rate,
                "tokenize_vocabsize": root_args.tokenizer.vocab_size
            }, root_args.output)
    elif kwargs['mode'] == 'length':
        dataset_dump(
            input_lens, output_lens, delays, {
                "workload_type": "splitwise",
                "num_requests": kwargs['num_requests'],
                "delay_distr": root_args.time_delay_dist,
                "request_rate": root_args.request_rate,
                "tokenize_vocabsize": root_args.tokenizer.vocab_size
            }, root_args.output, mode='length')
    else:
        for input_len in input_lens:
            input_ids.append("hi " * input_len)
        # make sure the tokenized input_ids are of the same length as the input_lens
        # for i in range(len(input_ids)):
        #     id = root_args.tokenizer.encode(input_ids[i])
        #     print(f"text len {input_lens[i]}, token len {len(id)}")
            
            
        dataset_dump(
            input_ids, output_lens, delays, {
                "workload_type": "splitwise",
                "num_requests": kwargs['num_requests'],
                "delay_distr": root_args.time_delay_dist,
                "request_rate": root_args.request_rate,
                "tokenize_vocabsize": root_args.tokenizer.vocab_size
            }, root_args.output, mode='text')
        
