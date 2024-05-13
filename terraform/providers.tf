terraform {
  required_version = ">= 0.14"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.118.0"
    }
  }
}

provider "yandex" {
  service_account_key_file = var.service_account_key_file
  cloud_id = var.cloud_id
  folder_id = var.folder_id
}