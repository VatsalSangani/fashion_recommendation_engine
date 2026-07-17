name: Deploy Fashion RecSys to GCP Cloud Run

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    name: Deploy to GCP Cloud Run
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}

      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: fashion-recsys
          region: europe-west2
          source: .
          flags: >
            --port 8000
            --memory 2Gi
            --cpu 1
            --min-instances 0
            --max-instances 1
            --timeout 600
            --clear-base-image
            --allow-unauthenticated
