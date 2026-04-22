students = {}

# 이름과 점수를 입력받아 딕셔너리에 저장
# 입력 종료: 빈 줄(그냥 엔터)
while True:
    line = input().strip()
    if not line:
        break
    name, score = line.split()
    students[name] = int(score)

sum_score = 0

for score in students.values():
    sum_score += score

avg_score = sum_score / len(students) if students else 0
print(avg_score)