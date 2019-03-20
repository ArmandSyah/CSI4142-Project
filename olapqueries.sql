/*
    1. Roll up by Location
*/
SELECT l.street_name_highway, COUNT(accident_key) AS TotalAccidentNumber
FROM "Accident Facts" af
    INNER JOIN "Accidents" a ON a.key = af.accident_key
    INNER JOIN "Locations" ; on l.key = af.location_key
GROUP BY l.street_name_highway
ORDER BY TotalAccidentNumber desc

SELECT l.neighbourhood, COUNT(accident_key) AS TotalAccidentNumber
FROM "Accident Facts" af
    INNER JOIN "Accidents" a ON a.key = af.accident_key
    INNER JOIN "Locations" l on l.key = af.location_key
GROUP BY l.neighbourhood
ORDER BY TotalAccidentNumber desc

SELECT l.city, COUNT(accident_key) AS TotalAccidentNumber
FROM "Accident Facts" af
    INNER JOIN "Accidents" a ON a.key = af.accident_key
    INNER JOIN "Locations" l on l.key = af.location_key
GROUP BY l.city
ORDER BY TotalAccidentNumber desc

/*
    2. Drill down by Date
*/

SELECT h.year, COUNT(accident_key) AS TotalAccidentNumber
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key= a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
WHERE l.city = 'ottawa'
 GROUP BY h.year;
 
SELECT h.month, COUNT(accident_key) AS TotalAccidentNumber
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key= a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
WHERE l.city = 'ottawa'
 GROUP BY h.month;
 
SELECT h.day, COUNT(accident_key) AS TotalAccidentNumber
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key= a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
WHERE l.city = 'ottawa'
 GROUP BY h.day;
 
SELECT h.hour_start, COUNT(accident_key) AS TotalAccidentNumber
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key= a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
WHERE l.city = 'ottawa'
 GROUP BY h.hour_start;

 /*
    3. Slice 
 */

 /* Slice: Explore the number of accidents in Orleans over the years*/

 Select h.year, COUNT(accident_key) AS TotalAccidentNumber
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key= a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
WHERE l.neighbourhood = 'Orleans'
GROUP BY h.year

/* Slice: Compare the number of accidents on Mondays, versus the number of accidents on Fridays.*/

 Select h.day_of_week, COUNT (accident_key)
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
 WHERE  (h.day_of_Week='MONDAY' OR h.day_of_week='FRIDAY')
 Group by h.day_of_week

 /* Slice: For instance, contrast the number of accidents in Nepean on Mondays between
3 and 9, with the number of fatalities in ByWard Market during the same period
of time. */

Select h.day_of_week, h.hour_start, l.neighbourhood, COUNT (accident_key)
 FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
 WHERE (l.neighbourhood='Nepean' OR l.neighbourhood='ByWard Market' ) AND 
		h.day_of_week='MONDAY' AND
		(h.hour_start BETWEEN 3 AND 9) 
 Group by h.day_of_week, h.hour_start, l.neighbourhood;

 /*
    Dice
 */

 /* Dice: Get the number of accidents in Ottawa, in the year 2014, when the weather was clear and road environment was Clear*/
 SELECT h.year, l.city, w.weather, a.environment , COUNT(accident_key)
 	FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
    INNER JOIN "Weather" w
    ON w.key = af.weather_key
WHERE l.city = 'Ottawa' AND h.year = 2014 AND a.environment = 'Clear' AND w.weather = 'Clear'
GROUP BY h.year, l.city, w.weather, a.environment

 /* Dice: Get the number of fatal accidents in Ottawa, in the month of January, relative humidity was above 50 and road visibility is dark*/
  SELECT h.month, l.city, w.relative_humidity, a.visibility , COUNT(accident_key)
 	FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
    INNER JOIN "Weather" w
    ON w.key = af.weather_key
WHERE l.city = 'Ottawa' AND h.year = 2014 AND (a.visibility LIKE '%' || 'Dark' || '%') AND w.relative_humidity > 50 AND af.is_fatal = true
GROUP BY h.year, l.city, w.relative_humidity, a.environment

/* 
    5. TOP N
*/

/* TOP N: Get the top 5 neighbourhoods with most amount of accidents */
SELECT l.neighbourhood, COUNT(accident_key) as TotalNumberOfAccidents
    FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
    INNER JOIN "Weather" w
    ON w.key = af.weather_key
GROUP BY l.neighbourhood
ORDER BY TotalNumberOfAccidents DESC
FETCH FIRST 5 ROWS ONLY

/*TOP N: Get the worst (top 1) year for accidents*/
SELECT h.year, COUNT(accident_key) as TotalNumberOfAccidents
    FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
    INNER JOIN "Weather" w
    ON w.key = af.weather_key
GROUP BY h.year
ORDER BY TotalNumberOfAccidents DESC
FETCH FIRST 1 ROWS ONLY

/* 
    6. BOT N
*/

/*BOT N: Get the Bot 5 neighbourhoods in terms of accidents*/
SELECT l.neighbourhood, COUNT(accident_key) as TotalNumberOfAccidents
    FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
    INNER JOIN "Weather" w
    ON w.key = af.weather_key
GROUP BY l.neighbourhood
ORDER BY TotalNumberOfAccidents ASC
FETCH FIRST 5 ROWS ONLY

/*BOT N: Get the Bot 3 months in the year of 2015 in ottawa, in terms of accidents */
SELECT l.city, h.month, COUNT(accident_key) as TotalNumberOfAccidents
    FROM "Accident Facts" af
	INNER JOIN "Accidents" a
	 ON af.accident_key = a.key
	INNER JOIN "Hours" h
	 ON h.key= af.hour_key
	INNER JOIN "Locations" l
	 ON l.key= af.location_key
    INNER JOIN "Weather" w
    ON w.key = af.weather_key
WHERE l.city = 'Ottawa' AND h.year = 2015
GROUP BY l.city, h.month
ORDER BY TotalNumberOfAccidents ASC
FETCH FIRST 3 ROWS ONLY