def _best_permutation(self, sim_matrix: np.ndarray) -> List[int]:
        # Maximise sum of |cosine similarities| via the Hungarian algorithm (O(K^3)).
        # linear_sum_assignment minimises cost, so we negate the absolute similarities.
        _, col_ind = linear_sum_assignment(-np.abs(sim_matrix))
        return col_ind.tolist()

    def _align_single_pca(
        self,
        ref_pca: StaticPCA,
        cur_pca: StaticPCA
    ) -> StaticPCA:

        ref_loadings = ref_pca.results.loadings.copy()
        cur_loadings = cur_pca.results.loadings.copy()
        cur_scores = cur_pca.results.scores.copy()

        sim_matrix = self._component_similarity_matrix(
            ref_loadings,
            cur_loadings
        )

        perm = self._best_permutation(sim_matrix)

        cur_loadings = cur_loadings.iloc[:, perm]
        cur_scores = cur_scores.iloc[:, perm]

        cur_pca.results.eigvals = cur_pca.results.eigvals[perm]
        cur_pca.results.explained_var_ratios = cur_pca.results.explained_var_ratios[perm]
        cur_pca.results.eigvecs = cur_pca.results.eigvecs[:, perm]

        pc_names = [f"PC{i+1}" for i in range(self.n_components)]

        cur_loadings.columns = pc_names
        cur_scores.columns = pc_names

        for j, pc in enumerate(pc_names):

            signed_dot = np.dot(
                ref_loadings.iloc[:, j].values,
                cur_loadings.iloc[:, j].values
            )

            if signed_dot < 0:

                cur_loadings.iloc[:, j] *= -1
                cur_scores.iloc[:, j] *= -1
                cur_pca.results.eigvecs[:, j] *= -1

        cur_pca.results.loadings = cur_loadings
        cur_pca.results.scores = cur_scores

        return cur_pca
