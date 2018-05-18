Para compilar o código basta executar:
$: make

Para executar o programa basta executar:
$: ./wisard

Para alterar os parâmetros de entrada:
$: ./wisard [arquivo de treino csv] [arquivo de test csv] [tamanho do endereçamento da RAM] [bleaching] [seed]
Esses parâmetros são opicionais, porém a ordem deles é obrigatória.

Exemplo:
$: ./wisard train.csv test.csv 28 1

[arquivo de treino] = train.csv
[arquivo de teste] = test.csv
[tamanho do endereçamento da RAM] = 28
[bleaching ativado] = 1
