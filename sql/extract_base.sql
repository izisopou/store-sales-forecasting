WITH FilteredHolidayTable AS(
SELECT
  date,
  type,
  locale,
  locale_name,
  description
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
  {target_col} -- Expects a leading comma if provided (e.g. ', base.sales')
FROM `store-sales-project.store_sales.{table_name}` base
LEFT JOIN `store-sales-project.store_sales.stores` stores
  ON base.store_nbr = stores.store_nbr
LEFT JOIN `store-sales-project.store_sales.oil` oil
  ON base.date = oil.date  
)

SELECT
  base.*,
  -- Check if date is a holiday (National, or matching store's state/city)
  (holidays.description IS NOT NULL) AS is_holiday,
  holidays.description AS holiday_name
FROM BaseWithStoresAndOil AS base
LEFT JOIN FilteredHolidayTable AS holidays
  ON base.date = holidays.date
  AND (
    holidays.locale = 'National'
    OR (holidays.locale = 'Regional' AND base.state = holidays.locale_name)
    OR (holidays.locale = 'Local' AND base.city = holidays.locale_name)
  )
-- Resolve multiple holidays per date: prioritize National > Regional > Local (tie-break alphabetically)
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY base.id
    ORDER BY
      CASE holidays.locale
        WHEN 'National' THEN 1
        WHEN 'Regional' THEN 2
        WHEN 'Local' THEN 3
        ELSE 4
      END ASC,
      holidays.description ASC
  ) = 1