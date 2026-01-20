Performance question

On which standard MaxSAT benchmarks does the memetic EA outperform pure local search from the same codebase?

Parameter question

How do population size and mutation probability affect solution quality and runtime?

Hybridization question

Does adding a strong local search phase (“memetic”) meaningfully improve performance compared to a plain genetic algorithm in this implementation?


250 vars population 50
->כלל אצבע פרקטי לטיונינג בלי להתפלסף

אם הליטוש לכל פרט הוא כבד → 
𝑃
=
50
P=50 מצוין להתחלה.

אם הליטוש קל (מעט צעדים) → עדיף 
𝑃
≈
100
P≈100 (או 
0.4
𝑁
0.4N).
אליטיזם מתון: לשמור 1–3 אליטים, לא 10. אחרת 50 ייסגר מהר.