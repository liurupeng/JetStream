# # Copyright 2024 Google LLC
# #
# # Licensed under the Apache License, Version 2.0 (the 'License');
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an 'AS IS' BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.




import argparse
import json
import logging
import os
import pandas
import random

from jetstream.engine.token_utils import load_vocab

_MIN_LEN = 4
_MAX_PROMPT_LEN = 1024
_MAX_TOTAL_LEN = 2048

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def load_openorca_dataset_pkl(filepath, tokenizer):
  '''Read dataset.'''
  data = pandas.read_pickle(filepath)
  dataset = []
  num_filtered = 0
  num_total = 0
  for _, row in data.iterrows():
    num_total += 1
    prompt = row['input']
    len_prompt_tokens = len(tokenizer.encode(prompt))
    output = row['output']
    len_output_tokens = len(tokenizer.encode(output))
    if len_prompt_tokens < _MIN_LEN or len_output_tokens < _MIN_LEN:
      num_filtered += 1
      continue
    total_output_tokens = len_prompt_tokens + len_output_tokens
    if len_prompt_tokens > _MAX_PROMPT_LEN or total_output_tokens > _MAX_TOTAL_LEN:
      num_filtered += 1
      continue
    dataset.append(
      {
        'prompt': prompt,
        'output': output,
        'len_prompt_tokens': len_prompt_tokens,
        'len_output_tokens': len_output_tokens,
        'idx': num_total
        }
      )
  logging.info('Requests: total:%d filtered %d', num_total, num_filtered)
  return dataset

def sample_requests(dataset, num_samples):
  assert num_samples <= len(dataset)
  dataset = random.sample(dataset, num_samples)
  for i in range(len(dataset)):
    dataset[i]['idx'] = i
  return dataset

def get_tokenizer(tokenizer_path):
  '''Return a tokenizer.'''
  vocab = load_vocab(tokenizer_path)
  return vocab.tokenizer

def main(args: argparse.Namespace) -> None:
  '''Main function.'''
  print(args)
  random.seed(args.seed)
  default_tokenizer_path = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    'tokenizers',
    'tokenizer.llama2'
  )
  default_dataset_path = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    'datasets',
    'open_orca_gpt4_tokenized_llama.sampled_24576.pkl',
  )
  default_output_path = f'/tmp/benchmark_maxtext/output_{args.num_samples}.json'
  output_path = args.output_path or default_output_path
  dataset_path = args.dataset_path or default_dataset_path
  tokenizer_path = args.tokenizer_path or default_tokenizer_path

  assert os.path.exists(dataset_path), f'missing data file {dataset_path}'
  assert os.path.exists(tokenizer_path), f'missing tokenizer {tokenizer_path}'

  output_dir = os.path.dirname(output_path)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  logging.info('Reading datafile %s', dataset_path)
  logging.info('Using tokenizer %s', tokenizer_path)

  tokenizer = get_tokenizer(tokenizer_path)
  dataset = load_openorca_dataset_pkl(dataset_path, tokenizer)
  samples = sample_requests(dataset, num_samples=args.num_samples)

  with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(samples, output_file)
  logging.info('Wrote %d samples at %s', len(samples), output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate sample datasets.')
  parser.add_argument('--num-samples', type=int, default=9, help='Num samples')
  parser.add_argument('--seed', type=int, default=0, help='Seed for randomness')
  parser.add_argument('--output-path', type=str, help='Path to output file')
  parser.add_argument('--dataset-path', type=str, help='Path to input pkl file')
  parser.add_argument('--tokenizer-path', type=str, help='Path to tokenizer')
  parsed_args = parser.parse_args()
  main(parsed_args)
