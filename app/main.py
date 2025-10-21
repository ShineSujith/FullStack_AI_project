from play_ground import play

if __name__ == '__main__':
    year = int(input("Enter year: "))
    play("stocks", "plots", f"{year}_perf.csv", year)
