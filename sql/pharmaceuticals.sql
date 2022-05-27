CREATE TABLE pharmaceuticalData(
    id INT AUTO_INCREMENT,
    Date DATETIME,
    DayOfWeek INT,
    Day INT,
    Month INT,
    Year INT,
    DayOfYear INT,
    WeekOfYear INT,
    Sales FLOAT,
    Assortment VARCHAR(255),
    CompetitionDistance FLOAT,
    CompetitionOpenSinceMonth FLOAT,
    CompetitionOpenSinceYear FLOAT,
    Customers FLOAT,
    Open INT,
    Promo INT,
    Promo2 INT,
    Promo2SinceWeek FLOAT,
    Promo2SinceYear FLOAT,
    PromoInterval VARCHAR(255),
    SchoolHoliday INT,
    StateHoliday INT,
    Store INT,
    StoreType INT,
    PRIMARY KEY(id)
);