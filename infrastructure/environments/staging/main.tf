terraform {
  required_version = ">= 1.5.0"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.110.0"
    }
  }
}

# TODO: заполнить переменные под твой облачный аккаунт
provider "yandex" {
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.zone
}

module "network" {
  source = "../../modules/network"
  name   = "credit-staging"
}

module "kubernetes" {
  source     = "../../modules/kubernetes"
  name       = "credit-staging"
  network_id = module.network.network_id
  subnet_id  = module.network.subnet_id
}

module "storage" {
  source = "../../modules/storage"
  name   = "credit-staging"
}

module "monitoring" {
  source = "../../modules/monitoring"
  name   = "credit-staging"
}
