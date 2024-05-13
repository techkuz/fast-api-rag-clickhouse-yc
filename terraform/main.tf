# Создание random-string
resource "random_string" "random" {
  length              = 4
  special             = false
  upper               = false 
}


# Create a Yandex VPC
resource "yandex_vpc_network" "example-vpc" {
  name      = "example-vpc-${random_string.random.result}"
}

resource "yandex_vpc_subnet" "example-subnet" {
  folder_id           = var.folder_id
  count               = 3
  name                = "app-example-subnet-${element(var.network_names, count.index)}"
  zone                = element(var.zones, count.index)
  network_id          = yandex_vpc_network.example-vpc.id
  v4_cidr_blocks      = [element(var.app_cidrs, count.index)]
}

resource "yandex_vpc_default_security_group" "example-sg" {
  network_id = yandex_vpc_network.example-vpc.id

    ingress {
    description    = "HTTPS (secure)"
    port           = 8443
    protocol       = "TCP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description    = "clickhouse-client (secure)"
    port           = 9440
    protocol       = "TCP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description    = "Allow all egress cluster traffic"
    protocol       = "TCP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}


# Создание service account
resource "yandex_iam_service_account" "example-sa" {
  folder_id           = var.folder_id
  name                = "example-sa-${random_string.random.result}"
}

# Создание статического ключа для service account
resource "yandex_iam_service_account_static_access_key" "example-sa-sk" {
  service_account_id  = yandex_iam_service_account.example-sa.id
}

# Назначение прав на service account
resource "yandex_resourcemanager_folder_iam_binding" "admin" {
  folder_id           = var.folder_id
  role                = "admin"  # здесь стоит указать конкретные роли, а не общую admin

  members = [
    "serviceAccount:${yandex_iam_service_account.example-sa.id}",
  ]
}

# Create Yandex Object Storage bucket
resource "yandex_storage_bucket" "example-bucket" {
  bucket              = "example-bucket-${random_string.random.result}"
  access_key = yandex_iam_service_account_static_access_key.example-sa-sk.access_key
  secret_key = yandex_iam_service_account_static_access_key.example-sa-sk.secret_key
}

# Create Yandex ClickHouse Managed Database
resource "yandex_mdb_clickhouse_cluster" "example-cluster" {
  name      = "example-cluster"
  environment = "PRODUCTION"
  network_id = yandex_vpc_network.example-vpc.id
  security_group_ids = [yandex_vpc_default_security_group.example-sg.id]

  clickhouse {
    resources {
        resource_preset_id = "s2.small"
        disk_type_id = "network-ssd"
        disk_size = 32
    }
  }

  host {
    type = "CLICKHOUSE"
    zone = yandex_vpc_subnet.example-subnet[0].zone
    subnet_id = yandex_vpc_subnet.example-subnet[0].id
    assign_public_ip = true
  }

    database  {
    name = var.database_name
    }
  
  user {
    name     = var.clickhouse_user
    password = var.clickhouse_password
    permission {
        database_name = var.database_name
    }
  }
}

resource "yandex_datasphere_project" "rag-ds-project" {  # ваш СА, под которым вы запускате tf должен быть добавлен в созданное раннее community DS
  name = "iml-example-ds-project"
  description = "Datasphere Project description"
  community_id = var.datasphere_community_id

  settings = {
    subnet_id = yandex_vpc_subnet.example-subnet[0].id
    service_account_id = yandex_iam_service_account.example-sa.id
    commit_mode = "AUTO"
    security_group_ids = [yandex_vpc_default_security_group.example-sg.id]
    ide = "JUPYTER_LAB"
}