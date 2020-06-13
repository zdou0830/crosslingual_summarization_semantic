# -*- coding: utf-8 -*-
import logging

import six

import nltk

import torch
import torch.cuda
import random
import numpy as np

from beaver.data import build_dataset
from beaver.infer import beam_search, greedy_search, sampling
from beaver.loss import WarmAdam, LabelSmoothingLoss, MyLabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, printing_opt
from beaver.utils.metric import calculate_rouge

from beaver.sim.reload_models import load_sim_model
from beaver.sim.utils import Example, unk_string

import itertools
import collections

def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split() for _ in sentences]))

def my_union_lcs(evals, ref):
  lcs_union = set()
  ref_words = _split_into_words([ref])
  combined_lcs_length = 0
  for eval_s in evals:
    eval_ws = _split_into_words([eval_s])
    lcs = set(my_recon_lcs(ref_words, eval_ws))
    combined_lcs_length += len(lcs)
    lcs_union = lcs_union.union(lcs)

  union_lcs_count = len(lcs_union)
  union_lcs_value = union_lcs_count #/ combined_lcs_length if not combined_lcs_length == 0 else 0
  return union_lcs_value

def _lcs_table(ref, can):
  """Create 2-d LCS score table."""
  rows = len(ref)
  cols = len(can)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if ref[i - 1] == can[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table

def _backtrack_norec(t, ref, can):
  """Read out LCS."""
  i = len(ref)
  j = len(can)
  lcs = []
  while i > 0 and j > 0:
    if ref[i - 1] == can[j - 1]:
      lcs.insert(0, i-1)
      i -= 1
      j -= 1
    elif t[i][j - 1] > t[i - 1][j]:
      j -= 1
    else:
      i -= 1
  return lcs


def lcs_ind(ref, can):
  t = _lcs_table(ref, can)
  return _backtrack_norec(t, ref, can)

def _find_union(lcs_list):
  """Finds union LCS given a list of LCS."""
  return sorted(list(set().union(*lcs_list)))

def _union_lcs(ref, c_list):
  lcs_list = [lcs_ind(ref, c) for c in c_list]
  return [ref[i] for i in _find_union(lcs_list)]

def calc_rouge_L(candidate, reference):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    beta = 1.
    candidates = candidate.replace('.', '.\n')
    references = reference.replace('.', '.\n')
    def get_sents(text):
      sents = six.ensure_str(text).split("\n")
      sents = [x for x in sents if len(x)]
      return sents

    candidates = [ s.split() for s in get_sents(candidates)]
    references = [ s.split() for s in get_sents(references)]
    m = sum(map(len, references))
    n = sum(map(len, candidates))
    if m == 0 or n == 0:
      print(reference)
      print(references)
      return 0

    token_cnts_r = collections.Counter()
    token_cnts_c = collections.Counter()

    for s in references:
      token_cnts_r.update(s)
    for s in candidates:
      token_cnts_c.update(s)
    
    hits = 0
    for r in references:
      lcs = _union_lcs(r, candidates)
      for t in lcs:
        if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
          hits += 1
          token_cnts_c[t] -= 1
          token_cnts_r[t] -= 1
    rec = hits/m
    prec = hits/n
    return 2*prec*rec / (rec+prec+1e-12)

def calc_sim(model, cand_text, ref_vec):
  e = Example(cand_text)
  e.populate_embeddings(model.vocab, 1, 0)
  if len(e.embeddings) == 0:  
    e.embeddings.append(model.vocab[unk_string])
  wx1, wl1, wm1 = model.torchify_batch([e])
  vec = model.encode(wx1, wm1, wl1)
  return model.cosine(vec, ref_vec)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


def valid(model, criterion_task1, criterion_task2, valid_dataset, step):
    model.eval()
    total_n = 0
    total_task1_loss = total_task2_loss = 0.0
    task1_hypothesis, task1_references = [], []
    task2_hypothesis, task2_references = [], []

    def roul(hyps, refs):
      rl1 = 0
      for h, r in zip(hyps, refs):
        rl = calc_rouge_L(h, r)
        rl1 += rl
      return rl1 / len(hyps)

    for i, (batch, flag) in enumerate(valid_dataset):

        scores = model(batch.src, batch.tgt, flag)
        predictions = greedy_search(opt, model, batch.src, valid_dataset.fields, flag)

        if flag:  # task1
            task1_hypothesis += [valid_dataset.fields["task1_tgt"].decode_word(p) for p in predictions]
            task1_references += [valid_dataset.fields["task1_tgt"].decode_word(t) for t in batch.tgt]
        else:
            task2_hypothesis += [valid_dataset.fields["task2_tgt"].decode_word(p) for p in predictions]
            task2_references += [valid_dataset.fields["task2_tgt"].decode_word(t) for t in batch.tgt]

        total_n += 1

    bleu_task1 = calculate_bleu(task1_hypothesis, task1_references)
    bleu_task2 = calculate_bleu(task2_hypothesis, task2_references)
    rouge1_task1, rouge2_task1 = calculate_rouge(task1_hypothesis, task1_references)
    rouge1_task2, rouge2_task2 = calculate_rouge(task2_hypothesis, task2_references)
    mean_task1_loss = 0
    mean_task2_loss = 0
    rougel_task1 = roul(task1_hypothesis, task1_references) * 100
    rougel_task2 = roul(task2_hypothesis, task2_references) * 100
    logging.info("loss-task1: %.2f \t loss-task2 %.2f \t bleu-task1: %3.2f\t bleu-task2: %3.2f \t rouge1-task1: %3.2f \t rouge1-task2: %3.2f \t rouge2-task1: %3.2f \t rouge2-task2: %3.2f\t rougeL-task1: %3.2f \t rougeL-task2: %3.2f\t"
                 % (mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2, rougel_task1, rougel_task2))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2, rougel_task1, rougel_task2)


def train(sim_model, model, criterion_task1, criterion_task2, optimizer, train_dataset, valid_dataset, fields, alpha, beta):
    with torch.set_grad_enabled(False):
        valid(model, criterion_task1, criterion_task2, valid_dataset, optimizer.n_step)
    model.train()
    pad = fields["src"].pad_id
    pads = [fields["task1_tgt"].pad_id, fields["task2_tgt"].pad_id]
    total_task1_loss = total_task2_loss = 0.0
    model.zero_grad()
    diff_flag = True
    for i, (batch, flag) in enumerate(train_dataset):
        if optimizer.n_step > opt.max_train_step:
            break
        if flag:
            continue
        model.eval()    
        weights_sample = []
        weights_greedy = []
        one_tokens = batch.tgt[:, 0].view(-1, 1)
        with torch.no_grad():
            if diff_flag:
                samples = sampling(opt, model, batch.src, fields, flag)
                max_len = max(x.size()[0] for x in samples)
                for j, p in enumerate(samples):
                    a = torch.full([torch.tensor(max_len-p.size()[0])], fill_value=pads[flag]).long().to(device)
                    samples[j] = torch.cat([p, a])
                    if flag:
                        weights_sample.append(calc_rouge_L(train_dataset.fields["task1_tgt"].decode_word(p), train_dataset.fields["task1_tgt"].decode_word(batch.tgt[j])))
                    else:
                        preds, ori_preds = train_dataset.fields["task2_tgt"].decode_ori(p)
                        tgts, ori_tgts = train_dataset.fields["task2_tgt"].decode_ori(batch.tgt[j])
                        sim_score = calc_sim(sim_model, ori_preds, batch.en_vec[j].view(1, -1))
                        rouge_score = calc_rouge_L(preds, tgts) if beta > 0 else 0
                        weights_sample.append((1-beta)*sim_score+(beta)*rouge_score)
                samples = torch.stack(samples)
                samples = torch.cat([one_tokens, samples], -1)
            greeds = greedy_search(opt, model, batch.src, fields, flag)
            max_len = max(x.size()[0] for x in greeds)
            for j, p in enumerate(greeds):
                a = torch.full([torch.tensor(max_len-p.size()[0])], fill_value=pads[flag]).long().to(device)
                greeds[j] = torch.cat([p, a])
                if flag:
                    weights_greedy.append(calc_rouge_L(train_dataset.fields["task1_tgt"].decode_word(p), train_dataset.fields["task1_tgt"].decode_word(batch.tgt[j])))
                else:
                    preds, ori_preds = train_dataset.fields["task2_tgt"].decode_ori(p)
                    tgts = train_dataset.fields["task2_tgt"].decode_word(batch.tgt[j])
                    sim_score = calc_sim(sim_model, ori_preds, batch.en_vec[j].view(1, -1))
                    rouge_score = calc_rouge_L(preds, tgts) if beta > 0 else 0
                    weights_greedy.append((1-beta)*sim_score+(beta)*rouge_score)
            greeds = torch.stack(greeds)
            greeds = torch.cat([one_tokens, greeds], -1)
        if diff_flag:
            weights = (torch.Tensor(weights_sample) - torch.Tensor(weights_greedy)).to(device)
            scores = model(batch.src, samples, flag)
            model.train()
            if flag:
                loss = criterion_task1(scores, samples, weights)
            else:
                loss = criterion_task2(scores, samples, weights)
        else:
            weights = (torch.Tensor(weights_greedy)).to(device)
            scores = model(batch.src, greeds, flag)
            model.train()
            if flag:
                loss = criterion_task1(scores, greeds, weights)
            else:
                loss = criterion_task2(scores, greeds, weights)
        if alpha > 0 or not diff_flag:
          scores = model(batch.src, batch.tgt, flag)
        if diff_flag:
            if flag:
                loss = (1-alpha)*loss + alpha*criterion_task1(scores, batch.tgt) if alpha > 0 else loss
            else:
                loss = (1-alpha)*loss + alpha*criterion_task2(scores, batch.tgt) if alpha > 0 else loss
        else:
            if flag:
                loss += alpha*criterion_task1(scores, batch.tgt)
            else:
                loss += alpha*criterion_task2(scores, batch.tgt)
        loss.backward()

        if flag:  # task1
            total_task1_loss += loss.data
        else:
            total_task2_loss += loss.data

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_task1_loss = total_task1_loss / opt.report_every / opt.grad_accum * 2
                mean_task2_loss = total_task2_loss / opt.report_every / opt.grad_accum * 2
                logging.info("step: %7d\t loss-task1: %.4f \t loss-task2: %.4f"
                             % (optimizer.n_step, mean_task1_loss, mean_task2_loss))
                total_task1_loss = total_task2_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    valid(model, criterion_task1, criterion_task2, valid_dataset, optimizer.n_step)
                model.train()
        del loss


def main():
    sim_model = load_sim_model(opt.sim_model_file)

    logging.info("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields = train_dataset.fields
    logging.info("Build model...")

    pad_ids = {"src": fields["src"].pad_id,
               "task1_tgt": fields["task1_tgt"].pad_id,
               "task2_tgt": fields["task2_tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab),
                   "task1_tgt": len(fields["task1_tgt"].vocab),
                   "task2_tgt": len(fields["task2_tgt"].vocab)}

    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    criterion_task1 = MyLabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task1_tgt"], pad_ids["task1_tgt"]).to(device)
    criterion_task2 = MyLabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task2_tgt"], pad_ids["task2_tgt"]).to(device)

    n_step = 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("start training...")
    train(sim_model, model, criterion_task1, criterion_task2, optimizer, train_dataset, valid_dataset, fields, opt.alpha, opt.beta)


if __name__ == '__main__':
    main()
