terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.110.0"
    }
  }
}

variable "name" {
  type = string
}

variable "network_id" {
  type = string
}

variable "subnet_id" {
  type = string
}

# Заглушка под Managed Kubernetes (реальные ресурсы можно добавить позже)
output "cluster_name" {
  value = "${var.name}-k8s"
}
