class CausalTree:
    def __init__(self, df: pd.DataFrame, cause_names: List[str], effect_name: str):
        CausalTree.validate_causal_variables(df, cause_names, effect_name)
        self.df = df
        self.store = VariableStore()

        for name in cause_names + [effect_name]:
            self.store.add(name)

        self.type_level_causes: List[TypeLevelCause] = []
        self.type_level_significance_scores: List[CompositeScore] = []
        self.max_lag = -1

    @staticmethod
    def validate_causal_variables(df: pd.DataFrame, cause_names: List[str], effect_name: str):
        # check that all variables are in the dataframe
        missing_names = set(cause_names) - set(df.columns)
        if effect_name not in df.columns:
            missing_names.add(effect_name)
        if len(missing_names) > 0:
            raise ValueError(f"Variables not in df: {missing_names}")

        # check that all variables are binary-valued
        non_binary_cols = []
        for col in df.columns:
            if not df[col].isin([0, 1]).all():
                non_binary_cols.append(col)
        if len(non_binary_cols) > 0:
            raise ValueError(f"Non-binary columns: {non_binary_cols}")

    def get_variable_name(self, i: VariableIndex) -> str:
        return self.store.lookup_by_index(i)

    def get_variable_index(self, name: str) -> VariableIndex:
        return self.store.lookup_by_name(name)

    def get_relation(self, cause: str, effect: str) -> CausalRelation:
        cause_idx = self.store.lookup_by_name(cause)
        effect_idx = self.store.lookup_by_name(effect)
        return CausalRelation(cause_idx, effect_idx)

    def cause_holds_at(self, t: int, cause: Union[str, VariableIndex]) -> bool:
        # Returns true if the given cause is true at the given time step
        cause_name = cause if isinstance(cause, str) else self.get_variable_name(cause)
        return self.df.at[t, cause_name] == 1

    def effect_holds_in(self, window: Window, effect: Union[str, VariableIndex]) -> bool:
        # Returns true if the given effect happened in the given window
        e = effect if isinstance(effect, str) else self.get_variable_name(effect)
        return self.df.loc[window.start : window.end, e].sum() > 0

    def get_token_effect_times(self, relation: CausalRelation, window: Window, t: int) -> List[int]:
        # Returns time step(s) where causal effect is true in the given window
        w = Window(t + window.start, t + window.end)
        cause_holds = self.cause_holds_at(t, relation.cause)
        effect_holds_in_window = self.effect_holds_in(w, relation.effect)

        if cause_holds and effect_holds_in_window:
            effect_name = self.get_variable_name(relation.effect)
            # find each time step where effect is true
            subset = self.df.loc[w.start : w.end, effect_name]
            indices = subset[subset == 1].index
            effect_times = [t for t in indices]
            return effect_times
        return []

    def c_leadsto_e_in(self, window: Window, relation: CausalRelation, t: int) -> int:
        # Returns 1 if cause leads to effect inside window, 0 otherwise.
        w = Window(t + window.start, t + window.end)
        cause_holds = self.cause_holds_at(t, relation.cause)
        effect_holds_in_window = self.effect_holds_in(w, relation.effect)

        if cause_holds and effect_holds_in_window:
            return 1
        return 0

    @property
    def num_time_steps(self) -> int:
        return self.df.shape[0]

    def identify_potential_cause(
        self, relation: CausalRelation, window: Window
    ) -> Optional[TypeLevelCause]:
        # Finds type-level cause that precedes effect inside the window, if it exists.
        dt = window.end - window.start
        token_causes = []
        num_c_leadsto_e = 0

        for t in range(self.num_time_steps - dt):
            num_c_leadsto_e += self.c_leadsto_e_in(window, relation, t)
            effect_times = self.get_token_effect_times(relation, window, t)
            for t_effect in effect_times:
                token_causes.append(TokenCause(relation, t, t_effect))

        if num_c_leadsto_e > 0:
            cause_column = self.get_variable_name(relation.cause)
            prob = num_c_leadsto_e / self.df[cause_column].sum()
            return TypeLevelCause(relation, window, prob, token_causes)
        return None

    def build(self, effect: str, max_lag: int = 5, verbose: bool = False):
        # Build the tree
        if effect not in self.store:
            raise ValueError(f"Effect variable {effect} not in store")

        self.max_lag = max_lag  # maximum lag to consider
        # create all possible windows
        windows = []
        for start in range(1, max_lag):
            for end in range(start + 1, max_lag + 1):
                windows.append(Window(start, end))
        print(f"Created {len(windows)} windows.")
        if verbose:
            for window in windows:
                print(window)

        # identify all potential type-level causes
        for cause in self.store.names:
            if cause == effect:
                continue
            if verbose:
                print(f"Finding valid windows for {cause} => {effect}")
            relation = self.get_relation(cause, effect)
            num_causes = 0
            num_token_events = 0

            for window in windows:
                type_level_cause = self.identify_potential_cause(relation, window)
                if type_level_cause is not None:
                    self.type_level_causes.append(type_level_cause)
                    num_causes += 1
                    num_token_events += len(type_level_cause.token_events)
            if verbose:
                print(f"Found {num_causes} type-level relations for {cause} => {effect}")
                print(f"Found {num_token_events} token events for {cause} => {effect}")

    def _prob_given_c_and_x(self, c: TypeLevelCause, x: TypeLevelCause) -> np.array:
        # Computes $P(e\vert c \land x)$ for all lags
        window = c.window
        cause = self.get_variable_name(c.relation.cause)
        effect = self.get_variable_name(c.relation.effect)
        x_name = self.get_variable_name(x.relation.cause)
        dt = window.end - window.start
        num_e_after_cx = np.zeros(self.max_lag + 1)

        for t in range(self.num_time_steps - dt):
            # if cause or x did not happen, skip
            if self.df.at[t, cause] == 0 or self.df.at[t, x_name] == 0:
                continue
            for t1 in range(t + window.start, t + window.end + 1):
                if t1 >= self.num_time_steps:
                    break
                lag = t1 - t
                num_e_after_cx[lag] += self.df.at[t1, effect]

        if num_e_after_cx.sum() == 0:
            return np.zeros(self.max_lag + 1)

        num_c_and_x = self.df[(self.df[cause] == 1) & (self.df[x_name] == 1)].shape[0]
        if num_c_and_x == 0:
            return np.zeros(self.max_lag + 1)
        prob = num_e_after_cx / num_c_and_x
        return prob

    def _prob_given_notc_and_x(self, c: TypeLevelCause, x: TypeLevelCause) -> np.array:
        # Computes $P(e\vert \neg{c} \land x)$, for each lag
        window = c.window
        cause = self.get_variable_name(c.relation.cause)
        effect = self.get_variable_name(c.relation.effect)
        x_name = self.get_variable_name(x.relation.cause)
        dt = window.end - window.start
        num_e_after_notcx = np.zeros(self.max_lag + 1)

        for t in range(self.num_time_steps - dt):
            # if c happened or x did not happen, skip
            if self.df.at[t, cause] == 1 or self.df.at[t, x_name] == 0:
                continue
            for t1 in range(t + window.start, t + window.end + 1):
                if t1 >= self.num_time_steps:
                    break
                lag = t1 - t
                num_e_after_notcx[lag] += self.df.at[t1, effect]

        if num_e_after_notcx.sum() == 0:
            return np.zeros(self.max_lag + 1)

        num_notc_and_x = self.df[(self.df[cause] == 0) & (self.df[x_name] == 1)].shape[0]
        if num_notc_and_x == 0:
            return np.zeros(self.max_lag + 1)
        prob = num_e_after_notcx / num_notc_and_x
        return prob

    def filter_type_level_causes_by_window(self, window: Window) -> List[TypeLevelCause]:
        return [c for c in self.type_level_causes if c.window == window]

    def get_other_causes_in_window(self, c: TypeLevelCause) -> Set[TypeLevelCause]:
        return set(self.filter_type_level_causes_by_window(c.window)) - {c}

    def _compute_type_level_significance(self, cause: TypeLevelCause) -> np.array:
        e_avg = np.zeros(self.max_lag + 1)
        X = self.get_other_causes_in_window(cause)
        for x in X:
            e_avg += self._prob_given_c_and_x(cause, x) - self._prob_given_notc_and_x(cause, x)
        e_avg /= len(X)
        return e_avg

    def compute_significance(
        self,
        verbose: bool = False,
    ) -> List[CompositeScore]:
        #  Compute significance of each type-level cause
        scores = []
        for cause in self.type_level_causes:
            if verbose:
                print("Computing significance of type-level cause: %s" % cause)
            e_avg = self._compute_type_level_significance(cause)
            scores.append(CompositeScore(e_avg))

        self.type_level_significance_scores = scores
        return scores

    def prune(self):
        # Remove type-level causes with negative significance score from tree
        pruned_causes = []
        pruned_scores = []
        for cause, score in zip(self.type_level_causes, self.type_level_significance_scores):
            if score.score.sum() > 0:
                pruned_causes.append(cause)
                pruned_scores.append(score)
        self.type_level_causes = pruned_causes
        self.type_level_significance_scores = pruned_scores
