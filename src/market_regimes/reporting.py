"""Export aggregate diagnostics and readable market-state figures."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .features import MODEL_FEATURES, TRAIN_END, VALIDATION_END
from .evaluation import ordered_transition_table, expected_durations

PALETTE=["#2563eb","#0d9488","#d97706","#dc2626"]


def save_results(root, result):
    root=Path(root)
    destination=root/"results"
    destination.mkdir(exist_ok=True)
    tables={"hmm_selection":result["hmm"]["selection"],
            "hmm_selection_initializations":result["hmm"]["selection_initializations"],
            "hmm_refit_initializations":result["hmm"]["refit_initializations"],
            "kmeans_selection":result["km"]["selection"],
            "initialization_robustness":result["robustness"],"test_method_comparison":result["comparison"],
            "test_yearly_log_scores":result["yearly_scores"]}
    for name,table in tables.items(): table.to_csv(destination/f"{name}.csv",index=False)
    for name in ["fit_profiles","test_profiles"]:
        result[name].to_csv(destination/f"{name}.csv")
    transitions=ordered_transition_table(result["hmm"]["model"],result["names"])
    transitions.to_csv(destination/"transition_matrix.csv")
    durations=expected_durations(result["hmm"]["model"],result["names"])
    durations.to_csv(destination/"model_expected_durations.csv")
    manifest={**result["data_audit"],**result["feature_audit"]}
    (destination/"data_manifest.json").write_text(json.dumps(manifest,indent=2))
    summary=result["summary"]
    (destination/"summary.json").write_text(json.dumps(summary,indent=2))
    (root/"data/processed").mkdir(parents=True,exist_ok=True)
    result["timeline"].to_csv(root/"data/processed/daily_regimes.csv")
    artifact={"model":result["hmm"]["model"],"scaler":result["hmm"]["scaler"],"names":result["names"],
              "feature_names":MODEL_FEATURES,"fitted_through":VALIDATION_END,
              "last_refit_posterior":result["refit_probability"][-1],
              "expected_next_date":str(result["test"].index[0].date()),
              "feature_timing":"available after daily close"}
    (root/"artifacts").mkdir(exist_ok=True)
    joblib.dump(artifact,root/"artifacts/regime_hmm.joblib")


def make_figures(root,result):
    directory=Path(root)/"results/figures"
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","savefig.dpi":150})
    def save(fig,name):
        fig.savefig(directory/name,bbox_inches="tight")
        plt.close(fig)
    features=result["features"]
    test=result["test"]
    names=sorted(result["names"].values())
    color_map=dict(zip(names,PALETTE))

    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
    axes[0].plot(features.index,features.nifty50,color=PALETTE[0]); axes[0].set(ylabel="NIFTY 50",title="Input overview: historical index levels and volatility")
    axes[1].plot(features.index,features.volatility_20*100,color=PALETTE[1]); axes[1].set(ylabel="20-day realized\nvolatility (%)")
    axes[2].plot(features.index,features.india_vix,color=PALETTE[2]); axes[2].set(ylabel="India VIX",xlabel="Date")
    for ax in axes:
        ax.axvline(pd.Timestamp(TRAIN_END),color="#64748b",linestyle="--",linewidth=1)
        ax.axvline(pd.Timestamp(VALIDATION_END),color="#dc2626",linestyle="--",linewidth=1)
    fig.tight_layout()
    save(fig,"01_input_overview.png")

    selection=result["hmm"]["selection"]
    fig,ax=plt.subplots(figsize=(9,4))
    x=np.arange(len(selection))
    ax.bar(x-.18,selection.validation_mean_conditional_log_score,.36,color=PALETTE[0],label="HMM: conditional forward score")
    ax.bar(x+.18,selection.independent_gmm_validation_mean_log_score,.36,color=PALETTE[2],label="Independent Gaussian mixture")
    ax.set_xticks(x,[f"{k} states/components" for k in selection.states])
    ax.set(title="Validation density comparison, 2019–2021",ylabel="Mean log score (higher is better)")
    ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",frameon=False,fontsize=9)
    save(fig,"02_validation_selection.png")

    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(test.index,test.nifty50,color="#cbd5e1",linewidth=1,zorder=1)
    for name in names:
        mask=result["test_labels"].eq(name)
        ax.scatter(test.index[mask],test.loc[mask,"nifty50"],s=10,color=color_map[name],label=name,zorder=2)
    ax.set(title="Held-out market regimes, 2022–2025: forward-only HMM",ylabel="NIFTY 50 closing level",xlabel="Date")
    ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",title="Risk-ordered ID",frameon=False)
    save(fig,"03_heldout_market_regimes.png")

    window=result["test_probability_frame"].loc["2024-01-01":"2024-12-31"]
    fig,ax=plt.subplots(figsize=(12,4))
    ax.stackplot(window.index,[window[name] for name in names],labels=names,colors=[color_map[name] for name in names],alpha=.85)
    ax.set(title="Filtered state probabilities: 2024 held-out observations",ylabel="Conditional probability",xlabel="Date",ylim=(0,1))
    ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",frameon=False)
    save(fig,"04_filtered_probabilities.png")

    matrix=ordered_transition_table(result["hmm"]["model"],result["names"])
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap="Blues")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value=matrix.iloc[i,j]
            ax.text(j,i,f"{value:.3f}",ha="center",va="center",color="white" if value>.55 else "#172554")
    ax.set_xticks(range(len(matrix)),matrix.columns); ax.set_yticks(range(len(matrix)),matrix.index)
    ax.set(title="Fitted transition probabilities",xlabel="Next state",ylabel="Current state")
    fig.colorbar(im,ax=ax,label="Probability")
    save(fig,"05_transition_matrix.png")

    profile=result["test_profiles"].reindex(names)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    axes[0].bar(names,profile.share*100,color=[color_map[n] for n in names]); axes[0].set(title="Held-out state occupancy",ylabel="Percent of feature observations")
    axes[1].bar(names,profile.mean_trailing_volatility*100,color=[color_map[n] for n in names]); axes[1].set(title="Current trailing volatility",ylabel="Annualized percent")
    axes[2].bar(names,profile.mean_next_20_volatility*100,color=[color_map[n] for n in names]); axes[2].set(title="Subsequent 20-session volatility",ylabel="Annualized percent")
    fig.tight_layout()
    save(fig,"06_test_state_profiles.png")

    comparison=result["comparison"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(comparison.method,comparison.switch_rate*100,color=PALETTE[:len(comparison)])
    axes[0].set(title="Held-out hard-label switching",ylabel="Percent of consecutive observation pairs")
    axes[1].scatter(result["robustness"].seed,result["robustness"].test_ari_to_selected_fit,color=PALETTE[0],s=60)
    axes[1].set(title="HMM initialization sensitivity",ylabel="Test ARI to selected fit",xlabel="Training initialization seed",ylim=(-.05,1.05))
    axes[1].axhline(1,color="#cbd5e1",linewidth=1)
    fig.tight_layout()
    save(fig,"07_persistence_and_robustness.png")
