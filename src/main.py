import explore
import train
import evaluate
from preprocess import get_data

explore.run()
X_tr, X_te, y_tr, y_te, cols = get_data()
model = train.train(X_tr, y_tr, cols)
evaluate.evaluate(model, X_tr, X_te, y_tr, y_te)
print("\nDone. Plots and results are saved in the outputs/ folder.")  
