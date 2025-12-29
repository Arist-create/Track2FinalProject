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

# Заглушка под Object Storage (реальные ресурсы можно добавить позже)
output "bucket_name" {
  value = "${var.name}-bucket"
}
