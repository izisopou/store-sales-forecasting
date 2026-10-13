WITH FilteredHolidayTable AS(
SELECT
  date,
  type,
  locale,
  locale_name
FROM `store-sales-project.store_sales.holidays_events`
WHERE 
  type <> 'Work Day'
  AND transferred = False
  AND date BETWEEN '2013-01-01' AND '2017-08-31'
),

BaseWithStoresAndOil AS(
SELECT
  base.id,
  base.date,
  base.store_nbr,
  base.family,
  base.onpromotion,
  stores.city,
  stores.state,
  stores.type AS store_type,
  stores.cluster,
  oil.dcoilwtico AS oil_price
  {target_col}
FROM `store-sales-project.store_sales.{table_name}` base
LEFT JOIN `store-sales-project.store_sales.stores` stores
  ON base.store_nbr = stores.store_nbr
LEFT JOIN `store-sales-project.store_sales.oil` oil
  ON base.date = oil.date  
)

SELECT
  *,
  EXISTS(
    SELECT 1
    FROM FilteredHolidayTable AS holidays
    WHERE
      base.date = holidays.date
      AND (
        holidays.locale = 'National'
        OR (holidays.locale = 'Regional' AND base.state = holidays.locale_name)
        OR (holidays.locale = 'Local' AND base.city = holidays.locale_name)
      )
  ) AS is_holiday
FROM BaseWithStoresAndOil as base