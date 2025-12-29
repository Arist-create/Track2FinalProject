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

# Заглушка под мониторинг (реальные ресурсы можно добавить позже)
output "monitoring_name" {
  value = "${var.name}-monitoring"
}
