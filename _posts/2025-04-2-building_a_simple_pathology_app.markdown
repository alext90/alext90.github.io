---
layout: post
title:  "Part 0: Building a simple AI-based Pathology Web App"
date:   2025-04-2 15:33:15 +0200
categories: jekyll update
---
# Series: Building a simple AI-based Pathology Web App

I decided to write a simple series to document how I would build a simple AI-based pathology diagnostics app. The series will obviously not cover every aspect that is neccessary for building a full product-grade application and serves mainly for my personal learning process.
For now I have decided to focus on image-based data and found this [dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset/data). The dataset contains MRI images for three different brain tumor types.  

## Structure

I will seperate this series into the following parts:
- AI Development
- Web App
- Data Engineering
- UX/UI

## System Design
The following schema shows an overview over the system design:

<div style="text-align: center">
    <img src="{{ '/assets/img/webapp_diagram.png' | relative_url }}" alt="System-Design" title="System Design" width="500"/>
</div>

- Frontend (HTML templates in FastAPI)
- Backend (FastAPI)
- AI Model (PyTorch Model)
- PostgreSQL DB
- Data Warehouse (some cloud bucket)

**Optional**
- Dagster & DBT for data pipeline and modeling
- MLOps for training, versioning and monitoring model (MLFlow)
- Business Analytics (Metabase) 