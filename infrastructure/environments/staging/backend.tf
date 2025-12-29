terraform {
  backend "s3" {
    # TODO: заполни своими значениями (Object Storage)
    bucket   = "CHANGE_ME_BUCKET"
    key      = "staging/terraform.tfstate"
    region   = "ru-central1"
    endpoint = "https://storage.yandexcloud.net"

    skip_region_validation      = true
    skip_credentials_validation = true
  }
}
