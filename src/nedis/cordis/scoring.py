import numpy as np


class Score2d():
    
    def __init__(self, metric0, metric1, metric0_separation=np.max, combine_metrics=None) -> None:
        self.metric0 = metric0
        self.metric1 = metric1
        self.metric0_separation = metric0_separation

        if combine_metrics == "f1":
            def combine_metrics(m1, m2):
                return 2 * (m1 * m2) / (m1 + m2)
        elif isinstance(combine_metrics, (float, int)):
            def combine_metrics(m1, m2):
                return (1 + combine_metrics**2) * (m1 * m2) / (combine_metrics**2 * m1 + m2)
        self.combine_metrics = combine_metrics

    def __call__(self, cluster, y, disruption_values):
        y0 = y[:, 0]
        y1 = y[:, 1]
        # y0_unique = np.unique(y[:, 0])
        y1_unique = np.unique(y[:, 1])
        
        separation_per_group = np.array([self.metric0(y0[y1 == yy], disruption_values[y1 == yy]) for yy in y1_unique])
        intra_group_separation = self.metric0_separation(separation_per_group)

        inter_group_separation_msk = y0 == cluster["reference_data"][0]
        inter_group_separation = self.metric1(
            y1[inter_group_separation_msk], 
            disruption_values[inter_group_separation_msk])

        if self.combine_metrics is None:
            return np.array((intra_group_separation, inter_group_separation))  
        else:
            return self.combine_metrics(intra_group_separation, inter_group_separation) 