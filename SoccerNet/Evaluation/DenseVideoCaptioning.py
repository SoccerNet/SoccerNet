import json
import zipfile
from tqdm import tqdm
import glob
import os

from SoccerNet.utils import getListGames
from SoccerNet.Evaluation.utils import LoadJsonFromZip, getMetaDataTask

import numpy as np


from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu as Bleuold
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import random
import string

from collections import defaultdict

class Bleu(Bleuold):

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)
            
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        
        return score, scores


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class DVCEvaluator(object):

    def __init__(self, SoccerNet_path, Predictions_path, prediction_file="results_caption.json", label_files="Labels-caption.json", tious=None, max_proposals=1000, split="test", version=6, window_size=30):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not SoccerNet_path:
            raise IOError('Please input a valid ground truth file.')
        if not Predictions_path:
            raise IOError('Please input a valid prediction file.')

        self.tious = tious
        self.max_proposals = max_proposals
        self.window_size = window_size
        _, _, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.dict_event = dict_event
        self.list_games = getListGames(split, task="caption")
        self.tokenizer = PTBTokenizer()

        self.soda_func = self.soda_c
        

        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        self.ground_truths = {}
        self.prediction = {}

        # infer name of the prediction_file
        if prediction_file == None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path, "*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    break

        for game in self.list_games:
            
            half_1, half_2 = self.load_file(root=SoccerNet_path, file=os.path.join(game, label_files), label_or_prediction=True)
            self.ground_truths[game, 1] = dict(  zip(["timestamps", "sentences"],zip(*half_1)))
            self.ground_truths[game, 2] = dict(  zip(["timestamps", "sentences"],zip(*half_2)))

            half_1, half_2 = self.load_file(root=Predictions_path, file=os.path.join(game, prediction_file), label_or_prediction=False)
            #TOCHANGE !!!!!!!!!!!!!!!!
            # path = os.path.join(SoccerNet_path, game, labels)
            # half_1, half_2 = self.load_file(path=path, label_or_prediction=True)
            self.prediction[game, 1] = [{"timestamp" : list(timestamp), "sentence" : caption} for timestamp, caption in half_1]
            self.prediction[game, 2] = [{"timestamp" : list(timestamp), "sentence" : caption} for timestamp, caption in half_2]
        
        self.ground_truths = [self.ground_truths]

    def load_file(self, root:str, file : str, label_or_prediction : bool):

        # load labels
        if zipfile.is_zipfile(root):
            data = LoadJsonFromZip(root, file)
        else:
            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)
        
        half_1 = []
        half_2 = []
        for annotation in data["annotations" if label_or_prediction else "predictions"]:

            time = annotation["gameTime"]
            event = annotation["label"]
            half = int(time[0])
            if event not in self.dict_event or half > 2:
                continue

            minutes, seconds = time.split(' ')[-1].split(':')
            time = ( int(seconds) + 60 * int(minutes)) 

            start, end = time - self.window_size //2, time + self.window_size //2 + self.window_size % 2

            if half == 1:
                half_1.append(((start, end), annotation["anonymized" if label_or_prediction else "comment"]))
            else:
                half_2.append(((start, end), annotation["anonymized" if label_or_prediction else "comment"]))

        half_1 = sorted(half_1, key=lambda x: x[0][0])
        half_2 = sorted(half_2, key=lambda x: x[0][0])
        

        return half_1, half_2

    def preprocess(self):
        self.gt_vids = self.get_gt_vid_ids()
        n_ref = len(self.ground_truths)
        p_spliter = [0]
        g_spliter = [[0] for i in range(n_ref)]
        times = {}
        cur_preds = {}
        cur_gts = [{} for i in range(n_ref)]
        for i, vid in enumerate(self.gt_vids): 
            cur_preds.update({j+p_spliter[-1]:[{"caption": remove_nonascii(p["sentence"])}] for j,p in enumerate(self.prediction[vid])})
            times[i] = [p["timestamp"] for p in self.prediction[vid]]
            p_spliter.append(p_spliter[-1] + len(times[i]))
            for n in range(n_ref):
                if vid not in self.ground_truths[n]: 
                    g_spliter[n].append(g_spliter[n][-1])
                    continue
                cur_gts[n].update({j+g_spliter[n][-1]:[{"caption": remove_nonascii(p)}] for j,p in enumerate(self.ground_truths[n][vid]["sentences"])})
                g_spliter[n].append(g_spliter[n][-1] + len(self.ground_truths[n][vid]["sentences"]))
        tokenize_preds = self.tokenizer.tokenize(cur_preds)
        tokenize_gts = [self.tokenizer.tokenize(j) for j in cur_gts]
        for i, vid in enumerate(self.gt_vids): 
            _p = [tokenize_preds[j] for j in range(p_spliter[i],p_spliter[i+1])]
            self.prediction[vid] = {"timestamps":times[i], "sentences":_p}
            for n in range(n_ref):
                if vid not in self.ground_truths[n]: continue
                _g = [tokenize_gts[n][j] for j in range(g_spliter[n][i],g_spliter[n][i+1])]
                self.ground_truths[n][vid]["sentences"] = _g

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1
        start, end = interval_2
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def get_gt_vid_ids(self):
        vid_ids = set()
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate_activitynet(self):
        print("Running ActivityNet Metrics...")
        scores = {}
        for tiou in self.tious:
            _scores = self.evaluate_tiou(tiou)
            for metric, score in list(_scores.items()):
                if metric not in scores:
                    scores[metric] = []
                scores[metric].append(score)
        scores = {f"{metric}" : np.mean(score_tiou) for metric, score_tiou in scores.items()}
        scores['Recall'] = []
        scores['Precision'] = []
        for tiou in self.tious:
            precision, recall = self.evaluate_detection(tiou)
            scores['Recall'].append(recall)
            scores['Precision'].append(precision)
        
        scores["Precision"] = np.mean(scores["Precision"])
        scores["Recall"] = np.mean(scores["Recall"])

        print('-' * 80)
        print("Average across all tIoUs")
        print('-' * 80)
        for metric in scores:
            print('| %s: %2.4f'%(metric, 100 * scores[metric]))
        
        return scores

    def evaluate_detection(self, tiou):
        gt_vid_ids = self.get_gt_vid_ids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set()
                pred_set_covered = set()
                num_gt = 0
                num_pred = 0
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    new_precision = float(len(pred_set_covered)) / (pred_i + 1) 
                    best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos 
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()
        
        unique_index = 0

        # video id to unique caption ids mapping
        vid2capid = {}
        
        cur_res = {}
        cur_gts = {}
        
        
        for vid_id in gt_vid_ids:
            
            vid2capid[vid_id] = []

            # If the video does not have a prediction, then we give it no matches
            # We set it to empty, and use this as a sanity check later on.
            if vid_id not in self.prediction:
                pass

            # If we do have a prediction, then we find the scores based on all the
            # valid tIoU overlaps.
            else:
                # For each prediction, we look at the tIoU with ground truth.
                for pred in self.prediction[vid_id]:
                    has_added = False
                    for gt in self.ground_truths:
                        if vid_id not in gt:
                            continue
                        gt_captions = gt[vid_id]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if self.iou(pred['timestamp'], caption_timestamp) > tiou:
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1
                                has_added = True

                    # If the predicted caption does not overlap with any ground truth,
                    # we should compare it with garbage.
                    if not has_added:
                        cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                        cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                        vid2capid[vid_id].append(unique_index)
                        unique_index += 1

        # Each scorer will compute across all videos and take average score
        output = {}
        for scorer, method in self.scorers:
            
            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}
            
            # call tokenizer here for all predictions and gts
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)
            
            # reshape back
            for vid in list(vid2capid.keys()):
                res[vid] = {index:tokenize_res[index] for index in vid2capid[vid]}
                gts[vid] = {index:tokenize_gts[index] for index in vid2capid[vid]}
            
            for i, vid_id in enumerate(gt_vid_ids):

                if len(res[vid_id]) == 0 or len(gts[vid_id]) == 0:
                    if type(method) == list:
                        score = [0] * len(method)
                    else:
                        score = 0
                else:
                    score, scores = scorer.compute_score(gts[vid_id], res[vid_id])
                    
                all_scores[vid_id] = score

            
            if type(method) == list:
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
            else:
                output[method] = np.mean(list(all_scores.values()))
        return output

    def soda_a(self, iou, scores):
        _, pairs = self.chased_dp_assignment(iou)
        r, c = (*zip(*pairs),)
        max_score = {metric : np.sum(value[r, c])for metric, value in scores.items()}
        return max_score

    def soda_b(self, iou, scores):
        # same as soda_a
        _, pairs = self.chased_dp_assignment(iou)
        r, c = (*zip(*pairs),)
        max_score = {metric : np.sum(value[r, c])for metric, value in scores.items()}
        return max_score

    def soda_c(self, iou, scores):
        max_score = {metric : self.chased_dp_assignment(iou*value)[0] for metric, value in scores.items()}
        return max_score

    def soda_d(self, iou, scores):
        max_score, pairs = self.chased_dp_assignment(iou)
        max_score = {metric : max_score for metric in scores}
        return max_score

    def chased_dp_assignment(self, scores):
        """ 
        Run dp matching
        Recurrence:  
            dp[i,j] = 
                max(dp[i-1,j], dp[i-1,j-1] + scores[i,j], dp[i,j-1])
        """
        M, N = scores.shape
        dp = - np.ones((M, N))
        path = np.zeros((M, N))

        def transition(i, j):
            if dp[i, j] >= 0:
                return dp[i, j]
            elif i == 0 and j == 0:
                state = [-1, -1, scores[i, j]]
            elif i == 0:
                state = [-1, transition(i, j-1), scores[i, j]]
            elif j == 0:
                state = [transition(i-1, j), -1, scores[i, j]]
            else:
                state = [transition(i-1, j), transition(i, j-1), transition(i-1, j-1) + scores[i, j]]
            dp[i, j] = np.max(state)
            path[i, j] = np.argmax(state)
            return dp[i, j]

        def get_pairs(i, j):
            p = np.where(path[i][:j+1] == 2)[0]
            if i != 0 and len(p) == 0:
                return get_pairs(i-1, j)
            elif i == 0 or p[-1] == 0:
                return [(i, p[-1])]
            else:
                return get_pairs(i-1, p[-1]-1) + [(i, p[-1])]
        N, M = scores.shape
        max_score = transition(N-1, M-1)
        pairs = get_pairs(N-1, M-1)
        return max_score, pairs

    def calc_iou_matrix(self, preds, golds):
        return np.array([[self.iou(pred, ct) for pred in preds["timestamps"]] for ct in golds['timestamps']])

    def calc_score_matrix(self, preds, golds):
        # Reformat to fit the input of pycocoevalcap scorers.
        p_sent, g_sent = preds["sentences"], golds["sentences"]
        res = {index: p for index, p in enumerate(p_sent)}
        gts = [{index: g for index in range(len(p_sent))} for i, g in enumerate(g_sent)]

        output = {}
        for scorer, method in self.scorers:
            scores = np.array([scorer.compute_score(gt, res)[1] for gt in gts])
            if type(method) == list:
                for m in range(len(method)):
                    output[method[m]] = scores[:, m, :]
            else:
                output[method] = scores

        return output

    def evaluate_soda(self):
        print(f"Running SODA...")
        tious = self.tious
        p_best = defaultdict(lambda : [[] for i in range(len(self.tious))])
        r_best = defaultdict(lambda : [[] for i in range(len(self.tious))])
        f_best = defaultdict(lambda : [[] for i in range(len(self.tious))])
        n_pred = []
        for vid in tqdm(self.gt_vids):
            _p = defaultdict(lambda : [[] for i in range(len(self.tious))])
            _r = defaultdict(lambda : [[] for i in range(len(self.tious))])
            _f = defaultdict(lambda : [[] for i in range(len(self.tious))])
            pred = self.prediction[vid]
            n_pred.append(len(pred["sentences"]))
            for gt in self.ground_truths:
                if vid not in gt:
                    continue
                gold = gt[vid]
                # create matrix
                _iou = self.calc_iou_matrix(pred, gold)
                scores = self.calc_score_matrix(pred, gold)
                for i, tiou in enumerate(tious):
                    iou = np.copy(_iou)
                    iou[iou <= tiou] = 0.0
                    max_score = self.soda_func(iou, scores)
                    (n_g, n_p) = iou.shape
                    for metric, value in max_score.items():
                        p, r = value/n_p, value/n_g
                        _p[metric][i].append(p)
                        _r[metric][i].append(r)
                        _f[metric][i].append(2 * p * r / (p + r) if p+r > 0 else 0)
                    
            best_idx = {metric : np.argmax(value, axis=1) for metric, value in _f.items()}
            for metric in best_idx:
                for i, tiou in enumerate(tious):
                    p_best[metric][i].append(_p[metric][i][best_idx[metric][i]])
                    r_best[metric][i].append(_r[metric][i][best_idx[metric][i]])
                    f_best[metric][i].append(_f[metric][i][best_idx[metric][i]])
        
        precision = {metric : np.mean(value, axis=1) for metric, value in p_best.items()}
        recall = {metric : np.mean(value, axis=1) for metric, value in r_best.items()}
        f1 = {metric : np.mean(value, axis=1) for metric, value in f_best.items()}
        print(f"avg. outputs: {np.mean(n_pred)}")

        final_scores = {}
        for metric in f1 : 
            final_scores[f"SODA_{metric}_precision"] = np.mean(precision[metric])
            final_scores[f"SODA_{metric}_recall"] = np.mean(recall[metric])
            final_scores[f"SODA_{metric}_f1"] = np.mean(f1[metric])
        
        print('-' * 80)
        print("SODA result")
        print('-' * 80)
        for scorer_name, score in final_scores.items():
            print(f'| {scorer_name}:{score*100:2.4f}')
        
        return final_scores



def evaluate(SoccerNet_path, Predictions_path, prediction_file="results_caption.json", label_files="Labels-caption.json", split="test", version=2, window_size=30, include_SODA=True):
    # evaluate the prediction with respect to some ground truth
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - label_files: name of the label files - by default "Labels-cpation.json"
    #   - split: split to evaluate from ["test", "challenge"]
    #   - version: version of dataset [1, 2]
    # Return:
    #   - dictionary of metrics

    evaluator = DVCEvaluator(SoccerNet_path, Predictions_path, prediction_file=prediction_file, label_files=label_files, tious=[0], split=split, version=version, window_size=window_size)
    result = evaluator.evaluate_activitynet()
    if include_SODA:
        evaluator.preprocess()
        result = {**result, **evaluator.evaluate_soda()}
    else:
        result = {**result}
    return result
