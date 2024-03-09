import numpy as np
import sns as sns

# Load all_features from the local file
all_features = np.load('all_features.npy', allow_pickle=True)

models = ["YU", "CMRxRecon(LAX)", "CMRxRecon(SAX)", "ACDC", "MR-ART", "CMRxMotion", "LDCTIQAG2023"]

import numpy as np
from ridgeplot import ridgeplot


for i in range(len(all_features)):
    all_features[i] = np.array(all_features[i]).flatten()


all_features_final = [all_features[1], all_features[2], all_features[3], all_features[0], all_features[4], all_features[5], all_features[6]]

print(all_features_final[0].shape)


fig = ridgeplot(samples=all_features_final, 
      bandwidth=200,          
      kde_points=np.linspace(-10000, 10000, 200),
      colorscale="viridis",
      colormode="row-index",
      labels=models, 
      spacing = 4 / 9,

)

fig.update_layout(
    # title="Minimum and maximum daily temperatures in Lincoln, NE (2016)",
    height=850,
    width=1100,
    font_size=16,
    plot_bgcolor="white",
    xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
    yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
    yaxis_title="Datasets",
    showlegend=True,
)

fig.show()

