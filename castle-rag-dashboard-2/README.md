# CASTLE RAG Dashboard package

This folder contains the Dash frontend/application package for the CASTLE RAG Dashboard. The old mock-data prototype has been replaced by the full evidence-retrieval application.

For the complete project README, architecture, model acknowledgements, demo GIF, and run instructions, see:

```text
../README.md
```

Main entrypoint:

```bash
python app.py
```

On Snellius, prefer the Slurm wrapper from the repository root:

```bash
cd ~/mma-2026
sbatch castle-rag-dashboard-2/scripts/submit_dashboard.job
```
