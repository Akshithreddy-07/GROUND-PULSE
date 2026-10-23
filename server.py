from __future__ import annotations
import json,math,os
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"ml"/"dataset.csv"
FEATURES=["rms","peak","dominant_freq","e_5_50","e_50_150","e_150_400"]
LABELS=["none","footsteps","vehicle","other"]
df=pd.read_csv(DATA); X=df[FEATURES]; y=df["label"]; groups=df["session"]
train_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(X,y,groups))
models={"tree":DecisionTreeClassifier(max_depth=4,min_samples_leaf=5,random_state=42),"forest":RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=3,random_state=42)}
metrics={}
for name,model in models.items():
 model.fit(X.iloc[train_idx],y.iloc[train_idx]); pred=model.predict(X.iloc[test_idx])
 metrics[name]={"accuracy":round(float(accuracy_score(y.iloc[test_idx],pred))*100,1),"f1":round(float(f1_score(y.iloc[test_idx],pred,average="macro",zero_division=0))*100,1),"report":classification_report(y.iloc[test_idx],pred,labels=LABELS,output_dict=True,zero_division=0),"confusion":confusion_matrix(y.iloc[test_idx],pred,labels=LABELS).tolist()}
 model.fit(X,y)
metrics["tree"]["name"]="Decision Tree"; metrics["forest"]["name"]="Random Forest"
metrics["rows"]=int(len(df)); metrics["testRows"]=int(len(test_idx)); metrics["features"]=FEATURES
metrics["classes"]={str(k):int(v) for k,v in y.value_counts().to_dict().items()}
metrics["split"]="GroupShuffleSplit by session · 75/25 · seed 42"
class Handler(SimpleHTTPRequestHandler):
 def send_json(self,obj,status=200):
  b=json.dumps(obj,allow_nan=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(b))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(b)
 def do_GET(self):
  path=urlparse(self.path).path
  if path=="/api/status": return self.send_json(metrics)
  if path=="/api/dataset": return self.send_json(df.to_dict(orient="records"))
  return super().do_GET()
 def do_POST(self):
  if urlparse(self.path).path!="/api/predict": return self.send_json({"error":"unknown route"},404)
  try:
   body=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0")))); row={k:float(body["features"][k]) for k in FEATURES}
   if not all(math.isfinite(v) for v in row.values()): raise ValueError("Features must be finite numbers")
   sample=pd.DataFrame([row],columns=FEATURES); tree=models["tree"]; forest=models["forest"]
   t=str(tree.predict(sample)[0]); r=str(forest.predict(sample)[0])
   tp=tree.predict_proba(sample)[0]; rp=forest.predict_proba(sample)[0]
   tc={str(c):float(v) for c,v in zip(tree.classes_,tp)}; rc={str(c):float(v) for c,v in zip(forest.classes_,rp)}
   return self.send_json({"label":r,"conf":max(rc.values()),"treePrediction":t,"rfPrediction":r,"treeConfidence":max(tc.values()),"rfConfidence":max(rc.values()),"treeProbabilities":tc,"forestProbabilities":rc})
  except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e: return self.send_json({"error":str(e)},400)
 def log_message(self,fmt,*args): print("%s - %s"%(self.log_date_time_string(),fmt%args))
if __name__=="__main__":
 os.chdir(ROOT); print("GroundPulse ready at http://127.0.0.1:8765"); print(f"Loaded {len(df)} labeled windows; session-aware model evaluation complete.")
 ThreadingHTTPServer(("127.0.0.1",8765),Handler).serve_forever()
