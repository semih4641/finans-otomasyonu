# Oracle Cloud Always Free dağıtımı

Bu kurulumda Telegram botu Oracle VM üzerinde sürekli çalışır. GitHub Actions
yalnızca yeni kodu VM'e aktarır ve Docker servisini yeniden başlatır. Botun
sanal işlem kayıtları, cooldown durumu, logu ve çalışma sırasında eğitilen
modeller `/opt/finans-otomasyonu/runtime` altında korunur.

## 1. Oracle VM oluşturma

1. Oracle Cloud hesabında **Compute > Instances > Create instance** yolunu açın.
2. Always Free kapsamındaki bir Ampere A1 şekli seçin. Bu bot için 1 OCPU ve
   6 GB RAM yeterli bir başlangıçtır.
3. İmaj olarak Ubuntu 24.04 seçin. Workflow SSH kullanıcısını `ubuntu` kabul eder.
4. Bir public IPv4 adresi atayın. Yalnız SSH (TCP 22) girişi gerekir; bot Telegram'a
   dışarı doğru bağlandığı için HTTP/HTTPS portu açmayın.
5. Oracle'ın oluşturduğu SSH private key'i güvenli bir yere indirin. Repoya
   eklemeyin ve sohbet içinde paylaşmayın.
6. Gelişmiş seçeneklerde cloud-init/user data alanına
   `deploy/oracle-cloud-init.yaml` içeriğini yapıştırın.
7. VM açıldıktan sonra cloud-init'in paketleri kurması için birkaç dakika bekleyin.

## 2. SSH bağlantısını doğrulama

İlk bağlantıda Oracle'ın gösterdiği public IP ve indirilen private key kullanılır.
Bağlantı başarılı olduktan sonra sunucuda aşağıdakileri doğrulayın:

```bash
docker --version
docker-compose --version
ls -ld /opt/finans-otomasyonu/runtime
```

Docker grup üyeliğinin geçerli olması için ilk cloud-init tamamlandıktan sonra
yeni bir SSH oturumu açmak gerekebilir.

## 3. GitHub Actions secret'ları

Repository **Settings > Secrets and variables > Actions** bölümünde aşağıdaki
repository secret'larını oluşturun:

| Secret | Değer |
| --- | --- |
| `OCI_HOST` | VM'in public IPv4 adresi |
| `OCI_SSH_PRIVATE_KEY` | Oracle VM için indirilen private key'in tam içeriği |
| `OCI_SSH_KNOWN_HOSTS` | VM'in doğrulanmış SSH known_hosts satırı |
| `BOT_TOKEN` | BotFather Telegram tokenı |
| `CHAT_ID` | Bildirimlerin gideceği Telegram sohbet kimliği |
| `ACCOUNT_SIZE` | Risk hesabında kullanılacak başlangıç tutarı |

Private key, bot tokenı ve diğer secret değerlerini issue, commit, workflow
dosyası veya Actions loguna yazmayın.

`OCI_SSH_KNOWN_HOSTS` değeri, VM'e ilk güvenli bağlantı doğrulandıktan sonra
yerel bilgisayarda `ssh-keyscan -H SUNUCU_IP` ile alınabilir. Çıktıdaki anahtar
parmak izini ilk SSH bağlantısında görülen parmak iziyle karşılaştırın.

## 4. İlk dağıtım

Workflow ana dala bir push geldiğinde veya **Actions > Oracle Cloud'a dagit >
Run workflow** seçildiğinde çalışır. İlk başarılı dağıtımdan sonra:

```bash
cd /opt/finans-otomasyonu/app
docker-compose ps
docker logs --tail 100 finans-otomasyonu
```

Container durursa Docker `restart: unless-stopped` ilkesiyle yeniden başlatır.
VM yeniden başladığında Docker servisi ve bot otomatik olarak ayağa kalkar.

## Güvenlik notları

- Repo herkese açıktır; hiçbir gizli değeri dosyalara commit etmeyin.
- Oracle Security List/NSG içinde yalnız kendi IP'nizden TCP 22 erişimi tercih edin.
- Bot gelen HTTP bağlantısı kullanmadığından 80, 443 veya özel uygulama portu açmayın.
- SSH private key'in yedeğini şifreli ve erişimi kısıtlı bir yerde tutun.
