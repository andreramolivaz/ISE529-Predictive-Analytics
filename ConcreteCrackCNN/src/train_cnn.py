#!/usr/bin/env python
"""
Concrete-Crack CNN – cast etichette a float32 (niente dtype=string)
Salva:
  history_loss.png, confusion_train.png, confusion_test.png, metrics.txt
"""

import argparse, pathlib, json, tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

# ── modello ──
def build_model(shape):
    return models.Sequential([
        layers.Rescaling(1./255, input_shape=shape),
        layers.Conv2D(32,3,activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation="relu"), layers.Dropout(0.3),
        layers.Dense(1,activation="sigmoid")
    ])

def cm_plot(model, ds, tag, out):
    y_t, y_p = [], []
    for x,y in ds.unbatch():
        y_t.append(int(y))
        y_p.append(int(model.predict(tf.expand_dims(x,0),verbose=0)[0]>.5))
    cm = confusion_matrix(y_t,y_p)
    acc = (cm[0,0]+cm[1,1])/cm.sum()
    plt.figure(figsize=(4,3)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title(f"{tag} (acc={acc:.3f})")
    plt.savefig(out/f"confusion_{tag}.png"); plt.close()
    return acc, classification_report(y_t,y_p,output_dict=True)

def cast(ds):
    # qualunque dtype → float32  (0. → 1.)
    return ds.map(lambda x,y: (x, tf.cast(y, tf.float32)),
                  num_parallel_calls=tf.data.AUTOTUNE)

def main(a):
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    img=(150,150); batch=a.batch

    ds_tr = tf.keras.preprocessing.image_dataset_from_directory(
        a.data_dir, validation_split=.2, subset="training",
        seed=123, image_size=img, batch_size=batch, label_mode="int")
    ds_val= tf.keras.preprocessing.image_dataset_from_directory(
        a.data_dir, validation_split=.2, subset="validation",
        seed=123, image_size=img, batch_size=batch, label_mode="int")

    AUTOTUNE=tf.data.AUTOTUNE
    ds_tr = cast(ds_tr).prefetch(AUTOTUNE)
    ds_val= cast(ds_val).prefetch(AUTOTUNE)

    model = build_model((*img,3))
    model.compile(optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    cb = [
        callbacks.ReduceLROnPlateau(monitor="val_loss",
                                    factor=0.5, patience=1, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss",
                                patience=3, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(out/"best.h5", save_best_only=True,
                                  monitor="val_loss", verbose=1)
    ]

    hist = model.fit(ds_tr, epochs=a.epochs,
                     validation_data=ds_val, callbacks=cb, verbose=1)

    plt.figure(); plt.plot(hist.history["loss"]); plt.plot(hist.history["val_loss"])
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(["train","val"])
    plt.savefig(out/"history_loss.png"); plt.close()

    metrics={}
    for tag,ds in [("train",ds_tr),("test",ds_val)]:
        acc,rep = cm_plot(model, ds, tag, out)
        metrics[f"acc_{tag}"]=acc; metrics[f"report_{tag}"]=rep
    (out/"metrics.txt").write_text(json.dumps(metrics,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",required=True)
    p.add_argument("--out_dir",default="out")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch",type=int,default=32)
    main(p.parse_args())
