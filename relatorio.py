Logistic Regression
Logistic Regression é um modelo estatístico usado para problemas de classificação binária, como distinguir clientes bons de maus. Ele modela a probabilidade de uma observação pertencer a uma classe usando a função logística (sigmoide), que mapeia saídas lineares para valores entre 0 e 1. Esse método é amplamente utilizado em crédito scoring devido à sua interpretabilidade, permitindo analisar o impacto de cada variável (como renda ou histórico de pagamentos) na probabilidade de inadimplência. O modelo foi formalizado por David Cox em 1958 ("The Regression Analysis of Binary Sequences", Journal of the Royal Statistical Society), embora suas origens remontem ao trabalho de Pierre-François Verhulst no século XIX sobre crescimento populacional.

Decision Tree Classifier
Decision Trees são modelos de aprendizado de máquina que particionam os dados em subconjuntos homogêneos por meio de regras baseadas em features (ex.: "Renda > R$ 5.000?"). Cada divisão busca maximizar a pureza dos nós (usando métricas como entropia ou Gini), tornando-o eficaz para identificar padrões não lineares. Em classificação de clientes, árvores de decisão podem capturar interações complexas entre variáveis, como a combinação de idade e dívida atual. O algoritmo CART (Classification and Regression Trees), proposto por Breiman et al. em 1984 ("Classification and Regression Trees", Wadsworth), é uma das bases teóricas mais influentes.

Random Forest Classifier
Random Forest combina múltiplas árvores de decisão, cada uma treinada em subconjuntos aleatórios dos dados (bagging) e features, reduzindo overfitting. A classificação final é determinada por votação majoritária, oferecendo robustez a ruídos. Em análise de crédito, esse método identifica clientes de risco mesmo quando alguns preditores são irrelevantes, graças à sua capacidade de generalização. O trabalho seminal de Breiman (2001, "Random Forests", Machine Learning) estabeleceu seu fundamento teórico, destacando sua eficácia em problemas de alta dimensionalidade.



Gradient-Boosted Trees (GBTs) são um método avançado de aprendizado de máquina que utiliza uma técnica chamada boosting para combinar várias árvores de decisão simples em um modelo robusto e preciso. Diferentemente de algoritmos como Random Forest, que treinam árvores independentemente, o GBT treina árvores sequencialmente, onde cada nova árvore é ajustada para corrigir os erros das anteriores. O processo começa com uma previsão inicial básica, como a média da variável resposta, e, em seguida, calcula os resíduos (diferenças entre as previsões e os valores reais). A cada iteração, uma nova árvore é treinada para prever esses resíduos, e suas previsões são adicionadas ao modelo anterior com um peso controlado por uma taxa de aprendizado (learning rate), que evita ajustes excessivos. O algoritmo minimiza uma função de perda, como a log-loss para classificação, usando gradiente descendente, garantindo que cada etapa reduza progressivamente o erro. Essa abordagem é particularmente eficaz para problemas com dados desbalanceados, como na classificação de clientes inadimplentes, onde padrões sutis podem ser capturados pelas iterações sucessivas. O trabalho seminal de Jerome Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001), estabeleceu as bases teóricas desse método, destacando sua capacidade de lidar com relações complexas e não lineares nos dados.

Linear Support Vector Classifier (LinearSVC)

O Linear Support Vector Classifier (LinearSVC) é uma variação de Support Vector Machines (SVMs) projetada para classificação binária linear. Seu objetivo é encontrar o hiperplano ótimo que separa as classes com a maior margem possível, maximizando a distância entre os pontos de dados mais próximos de cada classe (vetores de suporte). Em problemas de classificação de clientes, como distinguir bons pagadores de maus pagadores, o LinearSVC busca uma fronteira de decisão linear que melhor divide grupos com base em features como renda, score de crédito ou histórico de pagamentos. Se os dados não forem linearmente separáveis, é possível usar técnicas como kernel tricks (embora o LinearSVC em si não os utilize diretamente), mas sua principal vantagem é a eficiência computacional em grandes conjuntos de dados. O método é fundamentado na teoria de SVMs desenvolvida por Vapnik e Chervonenkis na década de 1960, com contribuições posteriores de Cortes e Vapnik no artigo "Support-Vector Networks" (1995), que formalizou o uso de margens maximizadas para generalização robusta.

Naive Bayes

O classificador Naive Bayes é um modelo probabilístico baseado no Teorema de Bayes, que assume independência condicional entre as features (daí o termo "naive"). Apesar dessa simplificação, ele é surpreendentemente eficaz em muitas tarefas de classificação, incluindo filtragem de spam e análise de risco de crédito. Para classificar clientes, o modelo calcula a probabilidade posterior de um cliente ser "bom" ou "mau" com base em evidências como idade, renda e histórico de transações. Ele é especialmente útil quando há muitas features categóricas ou quando os dados são esparsos. A origem do método remonta ao trabalho de Maron (1961) sobre indexação automática, mas sua popularização veio com aplicações em processamento de linguagem natural e mineração de dados. Sua simplicidade e velocidade o tornam uma opção atraente para sistemas que exigem previsões rápidas, mesmo em cenários onde a suposição de independência não é estritamente válida.

Multilayer Perceptron (MLP)

O Multilayer Perceptron (MLP) é um tipo de rede neural artificial feedforward composta por múltiplas camadas de neurônios interconectados (input, hidden e output). Ele é capaz de aprender representações hierárquicas e não lineares dos dados, ajustando pesos sinápticos por meio de retropropagação (backpropagation). Em problemas de classificação de clientes, o MLP pode identificar interações complexas entre variáveis que modelos lineares ou baseados em árvores não capturam facilmente, como combinações não óbvias entre idade, tempo de emprego e tipo de conta bancária. O treinamento envolve a minimização de uma função de perda (ex.: entropia cruzada) usando otimizadores como SGD ou Adam. O artigo revolucionário de Rumelhart, Hinton e Williams (1986), "Learning Representations by Backpropagating Errors", estabeleceu as bases para o treinamento eficiente de redes neurais profundas. No entanto, MLPs exigem cuidados com overfitting (usando técnicas como dropout ou regularização L2) e são computacionalmente mais intensivos que modelos tradicionais.

Factorization Machines (FMClassifier)

As Factorization Machines (FMs) são modelos projetados para lidar com dados esparsos e interações entre features, usando fatorização de matrizes para reduzir dimensionalidade. Elas modelam relações lineares e de segunda ordem (interações entre pares de features) de forma eficiente, sendo ideais para sistemas de recomendação e classificação com variáveis categóricas de alta cardinalidade (ex.: IDs de produtos ou CEPs). Em scoring de crédito, FMs podem capturar padrões como "clientes com profissão X e idade Y tendem a ser inadimplentes" mesmo quando esses pares são raros nos dados. O método foi introduzido por Steffen Rendle em 2010 ("Factorization Machines", IEEE ICDM) e combina vantagens de SVMs com decomposição estilo collaborative filtering. Sua flexibilidade o torna útil em cenários onde features interagem de maneiras não lineares, mas sem a complexidade computacional de redes neurais profundas.

LightGBM

Desenvolvido pela Microsoft, o LightGBM é uma implementação otimizada de Gradient Boosting Machines (GBMs) que usa leaf-wise tree growth (crescimento por folha, em vez de nível por nível) e técnicas como Gradient-based One-Side Sampling (GOSS) para acelerar o treinamento em grandes conjuntos de dados. Ele é especialmente eficiente em problemas com milhares de features (ex.: dados transacionais de clientes) e lida bem com desbalanceamento de classes — comum em detecção de fraude ou inadimplência. O segredo de sua performance está no agrupamento inteligente de valores contínuos em histogramas e na priorização de folhas que mais reduzem a perda. O artigo "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., NIPS 2017) detalha suas inovações, como suporte nativo a GPU e paralelização. Empresas de fintechs e bancos adotam LightGBM por sua precisão superior e velocidade comparada a XGBoost em muitos cenários.

XGBoost

O XGBoost (eXtreme Gradient Boosting) é uma das implementações mais populares de gradient boosting, conhecida por sua eficiência e regularização avançada. Ele aprimora o GBM tradicional com features como penalidades L1/L2 (lasso/ridge), amostragem de features (column subsampling) e tratamento nativo de valores faltantes. Em classificação de clientes, ele se destaca em competições de ML (ex.: Kaggle) por sua capacidade de ajuste fino via early stopping e cross-validation. O modelo é baseado no artigo "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, SIGKDD 2016), que introduziu otimizações como sparsity-aware split finding para dados esparsos. Bancos usam XGBoost para prever risco de crédito com métricas como AUC-ROC, aproveitando sua interpretabilidade parcial (via feature importance) e suporte a múltiplas linguagens.

OneVsRest

OneVsRest (OvR) é uma estratégia meta-algorítmica que transforma problemas multiclasse em múltiplos classificadores binários. Por exemplo, para categorizar clientes em "baixo", "médio" e "alto" risco, treina-se um modelo binário para cada classe (ex.: "baixo vs. resto"), e a decisão final é a classe com maior confiança. Embora simples, pode ser combinado com modelos como Logistic Regression ou SVMs para extensão a cenários multiclasse. A técnica é baseada em trabalhos como "Reducing Multiclass to Binary" (Allwein et al., COLT 2000), que analisam decomposições de problemas multiclasse. Em aplicações reais, OvR é útil quando modelos nativamente multiclasse (ex.: Random Forest) não estão disponíveis ou são menos eficientes.
