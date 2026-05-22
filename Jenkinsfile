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
                // build
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('2. Test') {
            steps {
                echo "Running Automated Unit Tests..."
                // run pytest
                sh "docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} pytest test_agent.py"
            }
        }

        stage('3. Code Quality (SonarCloud)') {
            steps {
                echo "Running Static Application Security Testing (SAST) via SonarQube..."
                // sonarsource scanner
                sh """
                docker run --rm \
                    -v "${WORKSPACE}:/usr/src" \
                    sonarsource/sonar-scanner-cli \
                    -Dsonar.projectKey=stock-agent-devsecops \
                    -Dsonar.organization=${DOCKER_USERNAME} \
                    -Dsonar.host.url=https://sonarcloud.io \
                    -Dsonar.login=${SONAR_TOKEN}
                """
            }
        }

        stage('4. Security Scan (Trivy)') {
            steps {
                echo "Scanning Docker Image for High/Critical Vulnerabilities..."
                // aquasec trivy
                sh """
                docker run --rm \
                    -v /var/run/docker.sock:/var/run/docker.sock \
                    aquasec/trivy image --severity HIGH,CRITICAL --exit-code 0 ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        stage('5. Deploy (Staging)') {
            steps {
                echo "Deploying to Local Arch Staging Environment..."
                // stop, remove, & start
                sh """
                docker stop stock-agent-staging || true
                docker rm stock-agent-staging || true
                docker run -d --name stock-agent-staging -p 7860:7860 ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }

        stage('6. Release (Docker Hub)') {
            steps {
                echo "Pushing Versioned Release to Docker Hub..."
                // login & push
                sh """
                echo ${DOCKER_CREDS_PSW} | docker login -u ${DOCKER_CREDS_USR} --password-stdin
                docker push ${IMAGE_NAME}:${IMAGE_TAG}
                docker push ${IMAGE_NAME}:latest
                """
            }
        }

        stage('7. Monitoring (Health Check & Metrics)') {
            steps {
                echo "Simulating Live Monitoring and Uptime Check..."
                // sleep & check health & metrics
                sleep 5
                sh """
                echo "Checking Application Health..."
                curl -f http://localhost:7860/docs || exit 1
                echo "Checking Metrics Endpoint for Prometheus..."
                curl -f http://localhost:7860/metrics || echo "WARNING: /metrics not found, but app is up."
                """
            }
        }
    }

    post {
        always {
            // logout
            sh "docker logout || true"
        }
        success {
            echo "DevSecOps Pipeline Completed Successfully! Image is live at ${IMAGE_NAME}:${IMAGE_TAG}"
        }
        failure {
            echo "Pipeline Failed. Please check the logs."
        }
    }
}