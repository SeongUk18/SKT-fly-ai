pipeline {
    agent any
    parameters {
        choice(name: 'VERSION', choices: ['1.1.0', '1.2.0', '1.3.0'], description: '')
        booleanParam(name: 'executeTests', defaultValue: true, description: '')
    }
    stages {
        stage("Checkout") {
            steps {
                checkout scm
            }
        }
        stage("Build") {
            steps {
                dir('MLOps/git_practice/flask') {
                    sh 'docker-compose build web'
                }
            }
        }
        stage("Tag and Push") {
            steps {
                withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId: 'docker-hub', usernameVariable: 'DOCKER_USER_ID', passwordVariable: 'DOCKER_USER_PASSWORD']]) {
                    dir('MLOps/git_practice/flask') {
                        sh "docker tag jenkins-pipeline_web:latest ${DOCKER_USER_ID}/jenkins-app:${BUILD_NUMBER}"
                        sh "docker login -u ${DOCKER_USER_ID} -p ${DOCKER_USER_PASSWORD}"
                        sh "docker push ${DOCKER_USER_ID}/jenkins-app:${BUILD_NUMBER}"
                    }
                }
            }
        }
        stage("deploy") {
            steps {
                dir('MLOps/git_practice/flask') {
                    sh "docker-compose up -d"
                }
            }
        }
    }
}
