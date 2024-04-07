程式目的:
用超啟發式學習的演算法解決01背包問題，這回練習的是Genetic Algorithm。


實作方式:
於程式中建立了一個 GeneticAlgorithm 類別執行這次的模型訓練，以下分別介紹此類別中，各個方法的功能:
__init__() : 設定初始值。

generate_random_individual() : 隨機生成一個個體。

calculate_fitness() : 計算解法的適應度。

generate_initial_population() : 生成初始個體群體。

tournament_selection() : 選擇個體，挑選法為競爭法（tournament selection）。

crossover() : 交配操作。

mutation() : 突變操作，只有一成的機率會突變 。

genetic_algorithm() : 執行基因演算法，共100次迭代，並回傳收斂值。

plot_convergence() : 繪製收斂圖，x軸為迭代次數，y軸為收斂的值。


心得感想:
這個演算法比爬山演算法還要複雜，但設計的思路反而更吸引我。實作的部分，要是能再細心一點的話應該能更早寫出來。
