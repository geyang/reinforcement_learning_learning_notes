# Setting up Kubernetes on AWS

## Install `aws-cli`

```bash
pip install aws-cli
```

### Configure AWS CLI with your own credentials

for detailed instruction, take a look at [**cli getting started**](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html).
1. go to `aws` console, go to **Users**, create new user without password
2. download user access keys and secrets as `csv`.
3. then do
    ```bash
    aws configure # aws configure --profile user2
    ```
    and put in (the keys here are fake obviously).
    ```
    AWS Access Key ID [None]: AKIAI44QH8DHBEXAMPLE
    AWS Secret Access Key [None]: je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY
    Default region name [None]: us-west-1
    Default output format [None]: json
    ```
4. Now to **test credentials**, do
    ```bash
    aws ec2 describe-instances
    ```
    
## Using `kube-aws` to setup a Kubernetes cluster on AWS

### Install `kube-aws`

detailed instruction for this part is available at [**kuberneties on aws**](https://coreos.com/kubernetes/docs/latest/kubernetes-on-aws.html
    ). 

1. Go to the releases and download the latest release tarball for your architecture.
    
    https://github.com/kubernetes-incubator/kube-aws/releases

2. Extract the binary:
    ```bash
    tar zxvf kube-aws-${PLATFORM}.tar.gz
    Add kube-aws to your path:
    ```
3. now move the binary to usr/local/bin
    ```bash
    mv ${PLATFORM}/kube-aws /usr/local/bin
    ```
    
### Create AWS KMS key

Amazon KMS keys are used to encrypt and decrypt cluster TLS assets. If you already have a KMS Key that you would like to use, you can skip creating a new key and provide the Arn string for your existing key.

You can create a `KMS key` in the AWS console, or with the aws command line tool:

```bash
$ aws kms --region=<your-region> create-key --description="kube-aws assets"

{
    "KeyMetadata": {
        "CreationDate": 1458235139.724,
        "KeyState": "Enabled",
        "Arn": "arn:aws:kms:us-west-1:xxxxxxxxx:key/xxxxxxxxxxxxxxxxxxx",
        "AWSAccountId": "xxxxxxxxxxxxx",
        "Enabled": true,
        "KeyUsage": "ENCRYPT_DECRYPT",
        "KeyId": "xxxxxxxxx",
        "Description": "kube-aws assets"
    }
}
```

#### Tips
- To delete `kms` keys, you need to use `schedule-key-deletion`. This is because deleting `kms` keys could potentially be extremely distruptive. Therefore `aws` enforces a key-deletion grace-period.
    ```bash
    $ aws kms schedule-key-deletion --key-id ffb87608-39f7-4e1a-8348-f6a711799d00
    {
        "DeletionDate": 1497312000.0,
        "KeyId": "arn:aws:kms:us-east-1:500739060533:key/ffb87608-39f7-4e1a-8348-f6a711799d00"
    }
    ```
    
### Configure `kube-aws`

Now configure `kube-aws` with the `kms` credentials:

```bash
➜  mkdir kube-aws-assets
➜  cd kube-aws-assets
➜  kube-aws-assets kube-aws init \
--cluster-name=my-cluster-name \
--external-dns-name=my-cluster-endpoint \
--region=us-east-1 \
--availability-zone=us-east-1a \
--key-name=key-pair-name \
--kms-key-arn="<your kms key>"

Success! Created cluster.yaml
```

Genearte credentials
```bash
➜  kube-aws-assets kube-aws render credentials --generate-ca
Generating TLS credentials...
-> Generating new TLS CA
-> Generating new TLS assets
➜  kube-aws-assets kube-aws render credentials --ca-cert-path=./credentials/ca.pem --ca-key-path=./credentials/ca-key.pem
Generating TLS credentials...
-> Parsing existing TLS CA
-> Generating new TLS assets
➜  kube-aws-assets

```

Now render stack
```bash
➜  kube-aws-assets kube-aws render stack
Success! Stack rendered to ./stack-templates.

Next steps:
1. (Optional) Validate your changes to cluster.yaml with "kube-aws validate"
2. (Optional) Further customize the cluster by modifying templates in ./stack-templates or cloud-configs in ./userdata.
3. Start the cluster with "kube-aws up".
```

you can now take a look at the folder
```bash
➜  kube-aws-assets tree
.
├── cluster.yaml
├── credentials
│   ├── admin-key.pem
│   ├── admin.pem
│   ├── apiserver-key.pem
│   ├── apiserver.pem
│   ├── ca-key.pem
│   ├── ca.pem
│   ├── dex-key.pem
│   ├── dex.pem
│   ├── etcd-client-key.pem
│   ├── etcd-client.pem
│   ├── etcd-key.pem
│   ├── etcd.pem
│   ├── worker-key.pem
│   └── worker.pem
├── kubeconfig
├── stack-templates
│   ├── control-plane.json.tmpl
│   ├── node-pool.json.tmpl
│   └── root.json.tmpl
└── userdata
    ├── cloud-config-controller
    ├── cloud-config-etcd
    └── cloud-config-worker


```

### Rerender the Credentials and the Stack

```bash
kube-aws render credentials
kube-aws render stack
```
now 
```bash
kube-aws validate
```

then up
```bash
kube-aws-assets git:(master) ✗ kube-aws up --s3-uri s3://escherpad-kube-aws-bucket
```
and it gives
```
WARN: the worker node pool "nodepool1" is associated to a k8s API endpoint behind the DNS name "kube-test-cluster-endpoint" managed by YOU!
Please never point the DNS record for it to a different k8s cluster, especially when the name is a "stable" one which is shared among multiple k8s clusters for achieving blue-green deployments of k8s clusters!
kube-aws can't save users from mistakes like that
Creating AWS resources. Please wait. It may take a few minutes.
Success! Your AWS resources have been created:
Cluster Name:		kube-test
Controller DNS Names:	kube-test-APIEndpo-C11AITYLIE8J-58520729.us-east-1.elb.amazonaws.com

The containers that power your cluster are now being downloaded.

You should be able to access the Kubernetes API once the containers finish downloading.
```

### Connect `kubectl` client to cluster

This is the third step of the setup. To connect, visit the `credentials/` folder. According to [link](https://github.com/kubernetes-incubator/kube-aws/blob/0eef4f208737deaf665d9148dc2ebcaa2d339a48/Documentation/kubernetes-on-aws-render.md), 

> This directory contains both encryped and unencrypted TLS assets for your cluster, along with a pre-configured kubeconfig file which provides access to your cluster api via kubectl.
>
> You can also specify additional access tokens in tokens.csv as shown in the official docs.

## KubeCtl client tutorial

1. setup minikube
    - install virtual box
    - install minikube
        ```bash
        curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.20.0/minikube-darwin-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
        ```
2. Now you have two tutorials
    - This shorter tutorial for [**running kube locally with minikube**](https://kubernetes.io/docs/getting-started-guides/minikube/)
    - *and* this longer one that covers making your own docker container [**Hello Minikube**](https://kubernetes.io/docs/tutorials/stateless-application/hello-minikube/)

    - To access remote clusters, follow instruction [**access application cluster**](https://kubernetes.io/docs/tasks/access-application-cluster/service-access-application-cluster/)

3. **Monitoring and kube dashboard** To take a look at what's going on with your cluster, use the dashboard here [https://github.com/kubernetes/dashboard].
    
    or do:
    ```bash
    kubectl proxy
    ```
    then visit http://localhost:8001/ui and then you shall rejoice.

### To start local cluster with minikube

```bash
minikube start --vm-driver=xhyve
```