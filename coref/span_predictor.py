""" Describes SpanPredictor which aims to predict spans by taking as input
head word and context embeddings.
"""

from typing import List, Optional, Tuple

from coref.const import Doc, Span
import torch


class SpanPredictor(torch.nn.Module):
    def __init__(self, input_size: int, distance_emb_size: int, cluster_averaging: str):
        super().__init__()
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(input_size * 2 + 64, input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 4, 3, 1, 1),
            torch.nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.emb = torch.nn.Embedding(128, distance_emb_size) # [-63, 63] + too_far
        if cluster_averaging not in ['none', 'mean', 'maxpool']:
            raise ValueError(f"Expected cluster_averaging value to be one of 'none', 'mean', 'maxpool'.")
        self.cluster_averaging = cluster_averaging

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.ffnn.parameters()).device

    # def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
    #             doc: Doc,
    #             words: torch.Tensor,
    #             heads_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Calculates span start/end scores of words for each span head in
    #     heads_ids

    #     Args:
    #         doc (Doc): the document data
    #         words (torch.Tensor): contextual embeddings for each word in the
    #             document, [n_words, emb_size]
    #         heads_ids (torch.Tensor): word indices of span heads

    #     Returns:
    #         torch.Tensor: span start/end scores, [n_heads, n_words, 2]
    #     """
    #     # Obtain distance embedding indices, [n_heads, n_words]
    #     relative_positions = (heads_ids.unsqueeze(1) - torch.arange(words.shape[0], device=words.device).unsqueeze(0))
    #     emb_ids = relative_positions + 63               # make all valid distances positive
    #     emb_ids[(emb_ids < 0) + (emb_ids > 126)] = 127  # "too_far"

    #     # Obtain "same sentence" boolean mask, [n_heads, n_words]
    #     sent_id = torch.tensor(doc["sent_id"], device=words.device)
    #     same_sent = (sent_id[heads_ids].unsqueeze(1) == sent_id.unsqueeze(0))

    #     # To save memory, only pass candidates from one sentence for each head
    #     # pair_matrix contains concatenated span_head_emb + candidate_emb + distance_emb
    #     # for each candidate among the words in the same sentence as span_head
    #     # [n_heads, input_size * 2 + distance_emb_size]
    #     rows, cols = same_sent.nonzero(as_tuple=True)
    #     pair_matrix = torch.cat((
    #         words[heads_ids[rows]],
    #         words[cols],
    #         self.emb(emb_ids[rows, cols]),
    #     ), dim=1)

    #     lengths = same_sent.sum(dim=1)
    #     padding_mask = torch.arange(0, lengths.max(), device=words.device).unsqueeze(0)
    #     padding_mask = (padding_mask < lengths.unsqueeze(1))  # [n_heads, max_sent_len]

    #     # [n_heads, max_sent_len, input_size * 2 + distance_emb_size]
    #     # This is necessary to allow the convolution layer to look at several
    #     # word scores
    #     padded_pairs = torch.zeros(*padding_mask.shape, pair_matrix.shape[-1], device=words.device)
    #     padded_pairs[padding_mask] = pair_matrix

    #     res = self.ffnn(padded_pairs) # [n_heads, n_candidates, last_layer_output]
    #     res = self.conv(res.permute(0, 2, 1)).permute(0, 2, 1) # [n_heads, n_candidates, 2]

    #     scores = torch.full((heads_ids.shape[0], words.shape[0], 2), float('-inf'), device=words.device)
    #     scores[rows, cols] = res[padding_mask]

    #     # Make sure that start <= head <= end during inference
    #     if not self.training:
    #         valid_starts = torch.log((relative_positions >= 0).to(torch.float))
    #         valid_ends = torch.log((relative_positions <= 0).to(torch.float))
    #         valid_positions = torch.stack((valid_starts, valid_ends), dim=2)
    #         return scores + valid_positions
    #     return scores
    
    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc: Doc,
                words: torch.Tensor,
                heads_ids: torch.Tensor,
                clusters: List[List[int]]) -> torch.Tensor:
        """
        Calculates span start/end scores of words for each span head in
        heads_ids

        Args:
            doc (Doc): the document data
            words (torch.Tensor): contextual embeddings for each word in the
                document, [n_words, emb_size]
            heads_ids (torch.Tensor): word indices of span heads

        Returns:
            torch.Tensor: span start/end scores, [n_heads, n_words, 2]
        """
        # Obtain distance embedding indices, [n_heads, n_words]
        relative_positions = (heads_ids.unsqueeze(1) - torch.arange(words.shape[0], device=words.device).unsqueeze(0))
        emb_ids = relative_positions + 63               # make all valid distances positive
        emb_ids[(emb_ids < 0) + (emb_ids > 126)] = 127  # "too_far"

        # Obtain "same sentence" boolean mask, [n_heads, n_words]
        sent_id = torch.tensor(doc["sent_id"], device=words.device)
        same_sent = (sent_id[heads_ids].unsqueeze(1) == sent_id.unsqueeze(0))

        # head word average
        # NOTE: remove these loops later
        cluster_averages = []
        for cluster in clusters:
            if self.cluster_averaging == 'mean':
                cluster_averages.append(torch.mean(words[cluster], 0))
            elif self.cluster_averaging == 'maxpool':
                cluster_averages.append(torch.max(words[cluster], 0).values)

        if self.cluster_averaging != 'none':
            cluster_averages = torch.stack(cluster_averages)


        row_to_cluster = []
        for index, cluster in enumerate(clusters):
            row_to_cluster.extend([index] * len(cluster))
        row_to_cluster = torch.tensor(row_to_cluster, device=words.device).long()
        

        # To save memory, only pass candidates from one sentence for each head
        # pair_matrix contains concatenated span_head_emb + candidate_emb + distance_emb
        # for each candidate among the words in the same sentence as span_head
        # [n_heads, input_size * 2 + distance_emb_size]
        rows, cols = same_sent.nonzero(as_tuple=True)

        if self.cluster_averaging == 'none':
            pair_matrix = torch.cat((
                words[heads_ids[rows]],
                words[cols],
                self.emb(emb_ids[rows, cols]),
            ), dim=1)
        else:
            pair_matrix = torch.cat((
                cluster_averages[row_to_cluster[rows]],
                words[cols],
                self.emb(emb_ids[rows, cols]),
            ), dim=1)

        lengths = same_sent.sum(dim=1)
        padding_mask = torch.arange(0, lengths.max(), device=words.device).unsqueeze(0)
        padding_mask = (padding_mask < lengths.unsqueeze(1))  # [n_heads, max_sent_len]

        # [n_heads, max_sent_len, input_size * 2 + distance_emb_size]
        # This is necessary to allow the convolution layer to look at several
        # word scores
        padded_pairs = torch.zeros(*padding_mask.shape, pair_matrix.shape[-1], device=words.device)
        padded_pairs[padding_mask] = pair_matrix

        res = self.ffnn(padded_pairs) # [n_heads, n_candidates, last_layer_output]
        res = self.conv(res.permute(0, 2, 1)).permute(0, 2, 1) # [n_heads, n_candidates, 2]

        scores = torch.full((heads_ids.shape[0], words.shape[0], 2), float('-inf'), device=words.device)
        scores[rows, cols] = res[padding_mask]

        # Make sure that start <= head <= end during inference
        if not self.training:
            valid_starts = torch.log((relative_positions >= 0).to(torch.float))
            valid_ends = torch.log((relative_positions <= 0).to(torch.float))
            valid_positions = torch.stack((valid_starts, valid_ends), dim=2)
            return scores + valid_positions
        return scores

    # NOTE: this is causing issues
    def get_training_data(self,
                          doc: Doc,
                          words: torch.Tensor
                          ) -> Tuple[Optional[torch.Tensor],
                                     Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """ Returns span starts/ends for gold mentions in the document. """
        head2span = sorted(doc["head2span"])
        if not head2span:
            return None, None

        # NOTE: filter head2span to only contain spans for which we have word_clusters
        # NOTE: because the model should not be trained on predicting spans for which it will never output a head word
        word_clusters_flat = [c for cluster in doc['word_clusters'] for c in cluster]
        head2span_filtered = [h2s for h2s in head2span if h2s[0] in word_clusters_flat]

        # heads, starts, ends = zip(*head2span)
        heads, starts, ends = zip(*head2span_filtered)
        heads = torch.tensor(heads, device=self.device)
        starts = torch.tensor(starts, device=self.device)
        ends = torch.tensor(ends, device=self.device) - 1
        
        # NOTE: this is causing issues, no access to the cluster list
        # NOTE: less clusters in 'word_clusters' compared to 'span_clusters'. Because some have been dropped. However head2span is not accounting for this dropping?
        return self(doc, words, heads, doc['word_clusters']), (starts, ends)
        # return None, (starts, ends)

    def predict(self,
                doc: Doc,
                words: torch.Tensor,
                clusters: List[List[int]]) -> List[List[Span]]:
        """
        Predicts span clusters based on the word clusters.

        Args:
            doc (Doc): the document data
            words (torch.Tensor): [n_words, emb_size] matrix containing
                embeddings for each of the words in the text
            clusters (List[List[int]]): a list of clusters where each cluster
                is a list of word indices

        Returns:
            List[List[Span]]: span clusters
        """
        if not clusters:
            return []

        # NOTE: possibility to use antecedent_ids instead? Or an average of all the head embeddings in a cluster?
        heads_ids = torch.tensor(
            sorted(i for cluster in clusters for i in cluster),
            device=self.device
        )

        scores = self(doc, words, heads_ids, clusters)
        starts = scores[:, :, 0].argmax(dim=1).tolist()
        ends = (scores[:, :, 1].argmax(dim=1) + 1).tolist()

        head2span = {
            head: (start, end)
            for head, start, end in zip(heads_ids.tolist(), starts, ends)
        }

        return [[head2span[head] for head in cluster]
                for cluster in clusters]
