from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
import torch

from .utils import has_overlapping, has_overlapping_nested


class BaseDecoder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass
    
    def update_id_to_classes(self, id_to_classes, gen_labels, batch_size):
        if self.config.labels_decoder is not None:
            if self.config.decoder_mode == 'prompt':
                new_id_to_classes = []
                cursor = 0
                for i in range(batch_size):
                    original = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
                    k = len(original)                    # how many labels belong to this example
                    mapping = {idx + 1: gen_labels[cursor + idx] for idx in range(k)}
                    new_id_to_classes.append(mapping)
                    cursor += k
                id_to_classes = new_id_to_classes
        return id_to_classes
    
    def greedy_search(self, spans, flat_ner=True, multi_label=False):
        if flat_ner:
            has_ov = partial(has_overlapping, multi_label=multi_label)
        else:
            has_ov = partial(has_overlapping_nested, multi_label=multi_label)

        new_list = []
        span_prob = sorted(spans, key=lambda x: -x[-1])

        for i in range(len(spans)):
            b = span_prob[i]
            flag = False
            for new in new_list:
                if has_ov(b[:-1], new):
                    flag = True
                    break
            if not flag:
                new_list.append(b)

        new_list = sorted(new_list, key=lambda x: x[0])
        return new_list
    # --- ADD near top-level helpers (outside class) ---
    def _len_span(self, a): return (a[1] - a[0] + 1)
    def _inter_len(self, a, b):
        left = max(a[0], b[0]); right = min(a[1], b[1])
        return max(0, right - left + 1)
    def _is_nested(self, a, b):
        return (a[0] >= b[0] and a[1] <= b[1]) or (b[0] >= a[0] and b[1] <= a[1])
    def _is_partial_overlap(self, a, b):
        return self._inter_len(a, b) > 0 and not self._is_nested(a, b)
    def _iou_token(self, a, b):
        inter = self._inter_len(a, b)
        if inter == 0: return 0.0
        union = self._len_span(a) + self._len_span(b) - inter
        return inter / union if union > 0 else 0.0

    # --- ADD inside class BaseDecoder ---
    def decode_with_algo(self, spans, flat_ner=True, multi_label=False, **kwargs):
        """
        Router thuật toán: spans = [(start_tok, end_tok, label, gen_label_or_None, score), ...]
        """
        algo = kwargs.get("decoding_algo", "greedy")
        if algo == "nms":
            return self.nms_search(spans, flat_ner=flat_ner, multi_label=multi_label,
                                iou_threshold=kwargs.get("iou_threshold", 0.5))
        if algo == "mwis" and flat_ner and not multi_label:
            return self.mwis_search(spans)  # MWIS: flat + single-label
        return self.greedy_search(spans, flat_ner, multi_label=multi_label)  # fallback gốc

    def nms_search(self, spans, flat_ner=True, multi_label=False, iou_threshold=0.5):
        spans_sorted = sorted(spans, key=lambda x: -x[-1])  # theo score giảm dần
        kept = []
        for cand in spans_sorted:
            (i, j, lab, _, s) = cand
            suppress = False
            for sel in kept:
                same_label = (lab == sel[2]) or (not multi_label)
                if not same_label: 
                    continue
                if flat_ner:
                    if self._iou_token((i, j), (sel[0], sel[1])) >= iou_threshold:
                        suppress = True; break
                else:
                    # nested mode: chỉ suppress partial-overlap (giữ fully-nested)
                    if self._is_partial_overlap((i, j), (sel[0], sel[1])):
                        suppress = True; break
            if not suppress:
                kept.append(cand)
        kept.sort(key=lambda x: x[0])
        return kept

    def mwis_search(self, spans):
        """ MWIS (interval scheduling with weights) cho FLAT; end inclusive. """
        if not spans: return []
        import bisect
        cands = sorted(spans, key=lambda x: (x[1], x[0]))  # sort by end, then start
        ends = [c[1] for c in cands]
        def find_prev(k):
            i_k = cands[k][0]                 # start
            idx = bisect.bisect_left(ends, i_k) - 1  # last end < i_k
            return idx
        n = len(cands)
        p = [find_prev(k) for k in range(n)]
        dp = [0.0]*(n+1); take = [False]*n
        for k in range(1, n+1):
            w = cands[k-1][-1]
            alt = dp[p[k-1]+1] + w
            if alt > dp[k-1]: dp[k]=alt; take[k-1]=True
            else: dp[k]=dp[k-1]
        res=[]; k=n
        while k>0:
            if take[k-1]: res.append(cands[k-1]); k=p[k-1]+1
            else: k-=1
        res.reverse()
        return res



class SpanDecoder(BaseDecoder):
    def decode(
        self,
        tokens,                      # list[list[str]]
        id_to_classes,               # dict[int,str]  or list[dict]
        model_output,                # (B, L, K, C) – raw logits
        flat_ner=False,
        threshold=0.5,
        multi_label=False,
        sel_idx=None,
        gen_labels=None,             # list[str] – labels generated by the span‑decoder
        num_gen_sequences=1,
        **kwargs
    ):
        """
        Parameters
        ----------
        sel_idx   : torch.LongTensor or None
            For *decoder_mode == 'span'* the `(B, M)` matrix that tells, for every
            kept embedding, which flat `(start*max_width + width)` span position
            it came from.  Padded elements contain ‑1.

        gen_labels : list[str] or None
            The labels returned by `generate_labels`.  Their order is the same
            as the order in which embeddings were fed to the decoder
            (`sel_idx` flattened row‑major).
        """
        B, L, K, C = model_output.shape
        probs = torch.sigmoid(model_output)
    
        span_label_maps = [{} for _ in range(B)]        # one dict per sample
        if self.config.decoder_mode == "span" and sel_idx is not None and gen_labels is not None:
            cursor = 0
            for b in range(B):
                valid_pos = (sel_idx[b] != -1)
                n = valid_pos.sum().item()
                if n:                                     # map only if we kept spans
                    flat_indices = sel_idx[b, valid_pos].tolist()
                    start_index = cursor * num_gen_sequences
                    # Extract labels for spans in this batch
                    span_labels = gen_labels[start_index : start_index + n * num_gen_sequences]
                    # Group labels: each span gets `num_gen_sequences` consecutive labels
                    labels_b = [
                        span_labels[i * num_gen_sequences : (i + 1) * num_gen_sequences]
                        for i in range(n)
                    ]
                    span_label_maps[b] = dict(zip(flat_indices, labels_b))
                cursor += n

        spans = []
        for i in range(B):
            probs_i = probs[i]
            id_to_class_i  = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            span_i = []

            s_idx, k_idx, c_idx = torch.where(probs_i > threshold)
            for s, k, c in zip(s_idx.tolist(), k_idx.tolist(), c_idx.tolist()):
                end = s + k + 1
                if end > len(tokens[i]):          # skip if span exceeds sentence
                    continue

                flat_idx = s * K + k              # match encoder's flatten rule

                # pick entity type
                if gen_labels is not None:
                    gen_ent_type = span_label_maps[i].get(flat_idx)
                else:
                    gen_ent_type = None
                ent_type = id_to_class_i[c + 1]   # (+1 because 0 is <pad>)

                span_i.append((s, s + k, ent_type, gen_ent_type, probs_i[s, k, c].item()))

            # span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            span_i = self.decode_with_algo(span_i, flat_ner, multi_label=multi_label, **kwargs)
            spans.append(span_i)

        return spans

class TokenDecoder(BaseDecoder):
    def get_indices_above_threshold(self, scores, threshold):
        scores = torch.sigmoid(scores)
        return [k.tolist() for k in torch.where(scores > threshold)]

    def calculate_span_score(self, start_idx, end_idx, scores_inside_i, start_i, end_i, id_to_classes, threshold):
        span_i = []
        for st, cls_st in zip(*start_idx):
            for ed, cls_ed in zip(*end_idx):
                if ed >= st and cls_st == cls_ed:
                    ins = scores_inside_i[st:ed + 1, cls_st]
                    if (ins < threshold).any():
                        continue
                    # Get the start and end scores for this span
                    start_score = start_i[st, cls_st]
                    end_score = end_i[ed, cls_st]
                    # Concatenate the inside scores with start and end scores
                    combined = torch.cat([ins, start_score.unsqueeze(0), end_score.unsqueeze(0)])
                    # The span score is the minimum value among these scores
                    spn_score = combined.min().item()
                    span_i.append((st, ed, id_to_classes[cls_st + 1], None, spn_score))
        return span_i

    def decode(self, tokens, id_to_classes, model_output, flat_ner=False, threshold=0.5, multi_label=False, **kwargs):
        model_output = model_output.permute(3, 0, 1, 2)
        scores_start, scores_end, scores_inside = model_output
        spans = []
        for i, _ in enumerate(tokens):
            id_to_class_i = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            span_scores = self.calculate_span_score(
                self.get_indices_above_threshold(scores_start[i], threshold),
                self.get_indices_above_threshold(scores_end[i], threshold),
                torch.sigmoid(scores_inside[i]),
                torch.sigmoid(scores_start[i]),
                torch.sigmoid(scores_end[i]),
                id_to_class_i,
                threshold
            )
            # span_i = self.greedy_search(span_scores, flat_ner, multi_label)
            span_i = self.decode_with_algo(span_scores, flat_ner, multi_label, **kwargs)
            spans.append(span_i)
        return spans