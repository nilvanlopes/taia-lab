# PRD

## 1. Problema
O pipeline resolve um problema de classificação supervisionada em dados sintéticos de crédito. A configuração base pode utilizar um batch size relativamente alto, o que pode reduzir a frequência de atualização dos pesos durante o treino e limitar a qualidade do ajuste do modelo.

## 2. Base do repositório
Pipeline utilizado: `src/taia_lab/pipelines/supervised_pipeline.py`  
Config base utilizada: `configs/exp04_pipeline.yaml`

## 3. Entrada e saída
- Entrada: arquivo YAML de configuração do experimento e dados sintéticos gerados com `make_classification`.
- Saída: métricas de treino e validação, modelo salvo em `models/`, scaler salvo em `artifacts/`, relatório salvo em `reports/` e execução registrada no MLflow local.

## 4. Abordagem técnica
O pipeline executa um fluxo supervisionado completo: ingestão de dados, preparação com `train_test_split` e `StandardScaler`, treinamento de um `TinyMLP`, avaliação com métricas de loss e acurácia, salvamento de artefatos e rastreamento no MLflow. Essa base faz sentido porque permite realizar uma alteração simples e controlada no modelo sem modificar a estrutura principal do pipeline.

## 5. Modificação proposta
Criar uma nova configuração derivada de `configs/exp04_pipeline.yaml` com o nome `exp04_pipeline_A1_bs32.yaml` e alterar apenas o parâmetro `train.batch_size` de `64` para `32`.
Adicione a nova configuração ao makefile, seguindo o padrão já existente.

## 6. Hipótese
Ao reduzir `batch_size`, o treinamento fará atualizações de pesos mais frequentes ao longo de cada época, o que pode melhorar o ajuste do modelo e aumentar `val_acc`, embora também possa deixar o treinamento menos estável.

## 7. Evidência esperada
Devem aparecer duas execuções comparáveis no MLflow local, uma da configuração original e outra da configuração modificada, com diferença explícita no parâmetro `train.batch_size`. Também devem ser gerados relatórios e artefatos distintos em `reports/`, `models/` e `artifacts/`, permitindo comparar `train_loss`, `val_loss` e `val_acc`.

## 8. Uso de IA Generativa
Foi utilizado o codex cli para a geração do arquivo de configuração seguindo o padrão existente no pipeline.