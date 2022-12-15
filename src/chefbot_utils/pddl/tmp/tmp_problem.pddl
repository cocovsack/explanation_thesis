
(define (problem tmp_problem) (:domain cooking)
  (:objects
        banana - ingredient
	blueberry - ingredient
	bowl - container
	chocolatechips - ingredient
	cocopuffs - ingredient
	do - action_type
	eatingspoon - tool
	egg - pastry
	jellypastry - pastry
	main - meal
	measuringcup - container
	microwave - appliance
	milk - liquid
	mixingspoon - tool
	muffin - pastry
	oats - ingredient
	pan - container
	peanutbutter - ingredient
	pie - pastry
	robot - robot
	roll - pastry
	salt - ingredient
	say - action_type
	side - meal
	sink - appliance
	stove - appliance
	strawberry - ingredient
	water - liquid
  )
  (:init 
	(handempty robot)
	(hypothetical main)
	(hypotheticalpastry side)
	(inappliance pan stove)
	(inappliance water sink)
	(instorage banana)
	(instorage blueberry)
	(instorage bowl)
	(instorage chocolatechips)
	(instorage cocopuffs)
	(instorage eatingspoon)
	(instorage egg)
	(instorage jellypastry)
	(instorage measuringcup)
	(instorage milk)
	(instorage mixingspoon)
	(instorage muffin)
	(instorage oats)
	(instorage peanutbutter)
	(instorage pie)
	(instorage roll)
	(instorage salt)
	(instorage strawberry)
	(inworkspace microwave)
	(inworkspace pan)
	(inworkspace sink)
	(inworkspace stove)
	(isbanana banana)
	(isblueberry blueberry)
	(isbowl bowl)
	(ischocolatechips chocolatechips)
	(iscocopuffs cocopuffs)
	(isdo do)
	(iseatingspoon eatingspoon)
	(isegg egg)
	(isjellypastry jellypastry)
	(ismeasuringcup measuringcup)
	(ismicrowave microwave)
	(ismilk milk)
	(ismixingspoon mixingspoon)
	(ismuffin muffin)
	(isoats oats)
	(ispan pan)
	(ispeanutbutter peanutbutter)
	(ispie pie)
	(isroll roll)
	(issalt salt)
	(issay say)
	(issink sink)
	(isstove stove)
	(isstrawberry strawberry)
	(iswater water)
  )
  (:goal (and
	(isfruityoatmeal main)
	(iscompletedoatmeal main)
	(iscompletedmuffin side)
	(iscompletedpastry side)
	(not (error main))))
)
