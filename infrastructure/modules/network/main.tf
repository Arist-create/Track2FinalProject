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

resource "yandex_vpc_network" "this" {
  name = "${var.name}-net"
}

resource "yandex_vpc_subnet" "this" {
  name           = "${var.name}-subnet"
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.this.id
  v4_cidr_blocks = ["10.10.0.0/24"]
}

output "network_id" {
  value = yandex_vpc_network.this.id
}

output "subnet_id" {
  value = yandex_vpc_subnet.this.id
}
