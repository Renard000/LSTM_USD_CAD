# # save_artifacts.py
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import OneHotEncoder

# # --- IMPORTANT ---
# # Ce script suppose que tu as encore en mémoire :
# # encoder, cs_x, cs_y (les objets utilisés pour entraîner le modèle)
# # Si tu as redémarré l'interpréteur, ré-exécute la portion d'entraînement
# # pour recréer encoder, cs_x, cs_y, puis lance ce script.

# # Exemple : si tu as encore ces objets, décommente et exécute les lignes suivantes
# # pour les sauvegarder. Sinon, ré-exécute ton script d'entraînement
# # jusqu'au point d'avoir encoder, cs_x, cs_y et la variable "x_train".

# # from ton_script_train import encoder, cs_x, cs_y

# # Pour l'exemple, on vérifie que les objets existent
# try:
#     encoder  # variable créée dans ta session d'entrainement
#     cs_x
#     cs_y
# except NameError:
#     raise RuntimeError("encoder, cs_x ou cs_y non trouvés dans la session. Ré-exécute ton entraînement et lance ce script ensuite.")

# # Sauvegarde en pickle
# with open("encoder.pkl", "wb") as f:
#     pickle.dump(encoder, f)

# with open("cs_x.pkl", "wb") as f:
#     pickle.dump(cs_x, f)

# with open("cs_y.pkl", "wb") as f:
#     pickle.dump(cs_y, f)

# print("Artifacts sauvegardés : encoder.pkl, cs_x.pkl, cs_y.pkl")
