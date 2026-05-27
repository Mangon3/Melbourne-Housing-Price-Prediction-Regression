# Developer Notes

This document contains essential guides and workflows for maintaining, securing, and deploying the Stock Agent application.

## Pushing to Production

Because our application architecture utilizes multiple hosting providers, any production updates **must** be pushed to both of our remote repositories.

1. **GitHub (`github`)**: Acts as our source of truth. Pushing here triggers our Jenkins CI/CD pipeline, SonarCloud analysis, and Vercel frontend deployments.
2. **Hugging Face (`hf`)**: Directly hosts our backend FastAPI container via Hugging Face Spaces.

### Workflow
When you have finished local development and committed your changes, run:

```bash
# 1. Push to GitHub (Triggers Vercel UI & Jenkins Pipeline)
git push github main

# 2. Push to Hugging Face (Updates the Live Backend API)
git push hf main
```
*Note: Always ensure your code passes local tests before pushing to both remotes.*

## Jenkins CI/CD Navigation

Jenkins handles our automated DevSecOps pipeline, which includes building the Docker images, running security scans, and simulating staging deployments.

- **Dashboard**: Access the Jenkins dashboard via your configured host (e.g., `http://localhost:8080`).
- **Pipeline View**: Click on the `Stock-Agent` job to see the history of all pipeline runs.
- **Viewing Logs**: Click on a specific build number (e.g., `#42`), then select **Console Output** from the left menu. This is where you can debug failed builds or view deployment statuses.
- **Rollback Behavior**: If the deployment stage fails, the Jenkinsfile contains a `try/catch` block that automatically runs `docker-compose down` to tear down the broken environment.

## SonarCloud Code Quality

SonarCloud is integrated into our workflow to monitor code quality and security. It scans the codebase automatically on pushes to GitHub.

- **Dashboard**: Navigate to [sonarcloud.io](https://sonarcloud.io/) and log in.
- **Project View**: Select your `Stock-Agent` repository.
- **Metrics**: From the overview, you can track:
  - **Reliability** (Bugs)
  - **Security** (Vulnerabilities)
  - **Maintainability** (Code Smells & Technical Debt)
  - **Coverage** (Test coverage percentage)
- **Quality Gate**: Always check that the project passes the "Quality Gate" status indicator at the top of the dashboard before considering a release stable.

## Accessing Trivy Security Reports

Trivy scans our compiled Docker images for OS-level and application dependency vulnerabilities.

1. **Pipeline Integration**: The Trivy scan runs as a dedicated stage in the Jenkins pipeline.
2. **Viewing the Report**: To see the results, go to the Jenkins build's **Console Output** and scroll to the `Security Scan (Trivy)` stage. The vulnerabilities list will be printed directly in the logs.
3. **Local Scanning**: If you want to verify security before pushing, you can run Trivy locally against your built image:
   ```bash
   trivy image stockagent-stock-agent:latest
   ```
