variable "folder_id" {
  description = "Yandex.Cloud Folder ID where resources will be created"
  default     = "" # yc config get folder-id
}

variable "datasphere_community_id" {
  description = "datasphere community ID"
  default = ""
}

variable "cloud_id" {
  description = "Yandex.Cloud ID where resources will be created"
  default     = "" # yc config get cloud-id
}

variable "service_account_key_file" {
  description = "Path to Yandex.Cloud service_account_key_file"
  default     = "/Users/User/project/terraform/key.json" # https://cloud.yandex.com/en-ru/docs/iam/concepts/users/service-accounts
}


variable "database_name" {
    default = "example-database" 
}

variable "clickhouse_user" {
    default = "example-user" 
}

variable "clickhouse_password" {
    default = "example-P@$$w0rd" 
}

variable "zones" {
  description = "Yandex.Cloud default Zone for provisoned resources"
  type        = list(string)
  default     = ["ru-central1-a", "ru-central1-b", "ru-central1-d"]
}

variable "network_names" {
  description = "Yandex Cloud default Zone for provisoned resources"
  type        = list(string)
  default     = ["a", "b", "d"]
}

variable "app_cidrs" {
  type        = list(string)
  default     = ["192.168.1.0/24", "192.168.50.0/24", "192.168.70.0/24"]
}
