WITH territoire as
    (SELECT * FROM public.gcms_territoire WHERE srid = 5490)

-- WITH territoire as
--     (SELECT * FROM public.gcms_territoire WHERE srid = 5490)

SELECT * FROM territoire
    where ST_Intersects(
    ST_MakeEnvelope(870150, 6616950, 870350, 6617200, 0), territoire.geometrie)


-- SELECT ST_MakeValid(ST_Force2D(st_setsrid(batiment.geometrie,2154))) AS geometry,
--        1 as presence
-- FROM batiment, territoire
-- WHERE (batiment.gcms_territoire = territoire.code)
--     AND batiment.geometrie  && ST_MakeEnvelope(870150, 6616950, 870350, 6617200, 0)
--     AND not gcms_detruit

-- UNION
-- SELECT ST_MakeValid(ST_Force2D(st_setsrid(reservoir.geometrie,2154))) AS geometry,
--        1 as presence
-- FROM reservoir, territoire
-- WHERE (reservoir.gcms_territoire = territoire.code)
--     AND reservoir.geometrie && ST_MakeEnvelope(870150, 6616950, 870350, 6617200, 0)
--     AND (reservoir.nature = 'Château d''eau' OR reservoir.nature = 'Réservoir industriel')
--     AND NOT gcms_detruit

-- END
-- ELSE
-- WITH territoire as
--     (SELECT * FROM public.gcms_territoire WHERE srid = 5490)
-- SELECT
