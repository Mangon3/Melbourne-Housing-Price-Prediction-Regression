pipeline {
    agent any

    environment {
        // configs
        DOCKER_USERNAME = "mangouh"
        IMAGE_NAME = "${DOCKER_USERNAME}/stock-agent"
        IMAGE_TAG = "v1.0.${BUILD_NUMBER}"
        
        // credentials
        DOCKER_CREDS = credentials('docker-hub-credentials')
        SONAR_TOKEN = credentials('SONAR_TOKEN')
    }

    stages {
        stage('1. Build') {
            steps {
                echo "Building Docker Image: ${IMAGE_NAME}:${IMAGE_TAG}..."
                sh "docker build --network=host -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest . | tee build_log.txt"
                archiveArtifacts artifacts: 'build_log.txt', allowEmptyArchive: true
            }
        }

        stage('2. Test (Unit & Integration)') {
            steps {
                echo "--- Running Automated Unit Tests (with Coverage) ---"
                sh """
                # Clean up old caches that might have root permissions
                docker run --rm -u root -v "${WORKSPACE}:/app" ${IMAGE_NAME}:${IMAGE_TAG} rm -rf /app/.pytest_cache /app/.coverage /app/coverage.xml

                # Run unit tests and generate XML coverage report for SonarCloud
                docker run --rm -u \$(id -u):\$(id -g) -e USER=jenkins -v "${WORKSPACE}:/app" ${IMAGE_NAME}:${IMAGE_TAG} pytest --cov=. --cov-report=xml:coverage.xml test/
                
                # Fix paths in coverage.xml to match SonarScanner's expected base directory
                sed -i 's|<source>/app</source>|<source>/usr/src</source>|g' coverage.xml
                """
                
                echo "--- Running Automated Integration Tests via Docker Compose ---"
                sh """
                # Copy .env from Jenkins home for API keys
                cp /var/lib/jenkins/.env .env || true
                
                export IMAGE_NAME=${IMAGE_NAME}
                export IMAGE_TAG=${IMAGE_TAG}
                # Prevent port collisions by temporarily bringing down staging
                export COMPOSE_PROJECT_NAME=staging_env
                docker-compose --env-file .env -f docker-compose.yml down || true
                
                export COMPOSE_PROJECT_NAME=test_env_${BUILD_NUMBER}
                
                # Start IaC with env file
                docker-compose --env-file .env -f docker-compose.yml up -d
                sleep 15
                
                # Run the integration tests inside the running app container, appending to the unit test coverage
                APP_CONTAINER=\$(docker-compose --env-file .env ps -q stock-agent)
                docker exec -u \$(id -u):\$(id -g) -e USER=jenkins -e GOOGLE_API_KEY=\$(grep GOOGLE_API_KEY .env | cut -d '=' -f2) \${APP_CONTAINER} pytest --cov=. --cov-append --cov-report=xml:coverage.xml test/test_chat_system.py
                
                # Fix paths in coverage.xml to match SonarScanner's expected base directory
                sed -i 's|<source>/app</source>|<source>/usr/src</source>|g' coverage.xml
                """
            }
            post {
                always {
                    sh """
                    export COMPOSE_PROJECT_NAME=test_env_${BUILD_NUMBER}
                    echo "--- Container Logs (stock-agent) ---"
                    docker-compose --env-file .env -f docker-compose.yml logs stock-agent || true
                    echo "--- End Container Logs ---"
                    docker-compose --env-file .env -f docker-compose.yml down || true
                    """
                }
            }
        }

        stage('3. Code Quality (SonarCloud)') {
            steps {
                echo "Running Static Application Security Testing (SAST) via SonarQube..."
                sh """
                docker run --rm --network=host \
                    -v "${WORKSPACE}:/usr/src" \
                    sonarsource/sonar-scanner-cli \
                    -Dsonar.projectKey=Mangon3_Stock-Agent \
                    -Dsonar.organization=mango80200782 \
                    -Dsonar.host.url=https://sonarcloud.io \
                    -Dsonar.login=${SONAR_TOKEN} \
                    -Dsonar.exclusions="test/**,.venv/**,src/app/**,src/components/**,src/lib/**,**/*.ts,**/*.tsx,**/*.css" \
                    -Dsonar.python.coverage.reportPaths="coverage.xml" \
                    -Dsonar.issue.ignore.multicriteria=e1,e2,e3 \
                    -Dsonar.issue.ignore.multicriteria.e1.ruleKey=text:S8565 \
                    -Dsonar.issue.ignore.multicriteria.e1.resourceKey=pyproject.toml \
                    -Dsonar.issue.ignore.multicriteria.e2.ruleKey=docker:S8544 \
                    -Dsonar.issue.ignore.multicriteria.e2.resourceKey=Dockerfile \
                    -Dsonar.issue.ignore.multicriteria.e3.ruleKey=docker:S8541 \
                    -Dsonar.issue.ignore.multicriteria.e3.resourceKey=Dockerfile \
                    -Dsonar.qualitygate.wait=true
                """
            }
        }

        stage('4. Security Scan (Trivy)') {
            steps {
                echo "Scanning Docker Image for High/Critical Vulnerabilities..."
                sh """
                docker run --rm --network=host \
                    -v /var/run/docker.sock:/var/run/docker.sock \
                    -v "${WORKSPACE}/.trivyignore:/.trivyignore" \
                    aquasec/trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        stage('5. Deploy (Staging IaC)') {
            steps {
                echo "Deploying to Staging Environment via IaC..."
                script {
                    try {
                        sh """
                        export IMAGE_NAME=${IMAGE_NAME}
                        export IMAGE_TAG=${IMAGE_TAG}
                        export COMPOSE_PROJECT_NAME=staging_env
                        docker-compose --env-file .env -f docker-compose.yml up -d
                        """
                    } catch (Exception e) {
                        echo "Deployment failed! Initiating rollback to previous state..."
                        sh """
                        export COMPOSE_PROJECT_NAME=staging_env
                        docker-compose --env-file .env -f docker-compose.yml down
                        # In a real environment, we would re-deploy the previous successful image tag here.
                        """
                        error("Deployment failed and rollback executed.")
                    }
                }
            }
        }

        stage('6. Release (Docker Hub & Git)') {
            steps {
                echo "Pushing Versioned Release to Docker Hub & Tagging Git..."
                sh """
                echo \${DOCKER_CREDS_PSW} | docker login -u \${DOCKER_CREDS_USR} --password-stdin
                docker push ${IMAGE_NAME}:${IMAGE_TAG}
                docker push ${IMAGE_NAME}:latest
                """
                
                sh """
                git config --global user.email "jenkins@localhost" || true
                git config --global user.name "Jenkins CI" || true
                git tag -a ${IMAGE_TAG} -m "Release ${IMAGE_TAG}" || true
                echo "Simulated Git Tag Push: git push origin ${IMAGE_TAG}"
                """
            }
        }

        stage('7. Monitoring (Health & Alerts)') {
            steps {
                echo "Simulating Live Monitoring and Alerting..."
                sleep 5
                sh """
                echo "Checking Application Health..."
                curl -f http://localhost:7860/docs || exit 1
                
                echo "Checking Metrics Endpoint for Prometheus..."
                curl -f http://localhost:7860/metrics || echo "WARNING: /metrics not found, but app is up."
                
                echo "Simulating incident alert to Slack/Email..."
                echo '{"text": "Deployment ${IMAGE_TAG} successful! Live on Staging."}' > alert.json
                echo "Webhook payload:"
                cat alert.json
                """
            }
        }
    }

    post {
        always {
            sh "docker logout || true"
        }
        success {
            echo "DevSecOps Pipeline Completed Successfully! Image is live at ${IMAGE_NAME}:${IMAGE_TAG}"
        }
        failure {
            echo "Pipeline Failed. Review the security and code quality gates."
        }
    }
}