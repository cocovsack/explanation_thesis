
(define (domain cooking)
      (:requirements :typing)
  (:types
    appliance moveable robot food location action_type - object
    ingredient container tool - moveable
    pastry liquid meal ingredient - food
  )
  (:predicates
  (dummyeffect ?x - object)
  (inworkspace ?v0 - object)
	(instorage ?v0 - moveable)
	(incontainer ?v0 - food ?v1 - container)
	(inappliance ?v0 - moveable ?v1 - appliance)
	(ison ?v0 - appliance)
	(occupied ?v0 - appliance)
	(issay ?v0 - action_type)
	(isdo ?v0 - action_type)
	(ispan ?v0 - container)
	(iswater ?v0 - liquid)
	(ismilk ?v0 - liquid)
	(isoats ?v0 - ingredient)
	(issalt ?v0 - ingredient)
	(isbowl ?v0 - container)
	(ismixingspoon ?v0 - tool)
	(iseatingspoon ?v0 - tool)
	(isbanana ?v0 - ingredient)
	(ischerry ?v0 - ingredient)
	(isstrawberry ?v0 - ingredient)
	(isblueberry ?v0 - ingredient)
	(ispeanutbutter ?v0 - ingredient)
	(ismeasuringcup ?v0 - container)
	(ischocolatechips ?v0 - ingredient)
	(isnuts ?v0 - ingredient)
	(ispie ?v0 - pastry)
	(ismuffin ?v0 - pastry)
	(isroll ?v0 - pastry)
	(isjellypastry ?v0 - pastry)
	(isegg ?v0 - pastry)
	(iscocopuffs ?v0 - ingredient)
	(iscereal ?v0 - meal)
	(ismicrowave ?v0 - appliance)
	(istoaster ?v0 - appliance)
	(isstove ?v0 - appliance)
	(issink ?v0 - appliance)
	(inhand ?v0 - moveable)
	(handempty ?v0 - robot)
	(mixed ?v0 - meal)
	(heated ?v0 - moveable)
	(warmed ?v0 - pastry)
	(boiling ?v0 - food)
	(simmering ?v0 - meal)
	(hypothetical ?v0 - meal)
	(error ?v0 - meal)
	(hypotheticalpastry ?v0 - meal)
	(isoatmeal ?v0 - meal)
	(isplainoatmeal ?v0 - meal)
	(isfruityoatmeal ?v0 - meal)
	(isnuttyoatmeal ?v0 - meal)
	(ischocolateoatmeal ?v0 - meal)
	(isfruitychocolateoatmeal ?v0 - meal)
	(ispeanutbutterbananaoatmeal ?v0 - meal)
	(ispeanutbutterchocolateoatmeal ?v0 - meal)
	(isnuttyfruityoatmeal ?v0 - meal)
	(isnuttychocolateoatmeal ?v0 - meal)
	(isnuttyfruitychocolateoatmeal ?v0 - meal)
	(isnuttypeanutbutterbananaoatmeal ?v0 - meal)
	(isnuttypeanutbutterchocolateoatmeal ?v0 - meal)
	(iscompletedoatmeal ?v0 - meal)
	(iscompletedcereal ?v0 - meal)
	(iscompletedpastry ?v0 - meal)
	(iscompletedmuffin ?v0 - meal)
	(iscompletedpie ?v0 - meal)
	(iscompletedjellypastry ?v0 - meal)
	(iscompletedegg ?v0 - meal)
	(iscompletedroll ?v0 - meal)
	(mix ?v0 - container ?v1 - meal ?v2 - action_type)
	(gather ?v0 - moveable ?v1 - action_type)
	(pour ?v0 - food ?v1 - container ?v2 - action_type)
	(pourwater ?v0 - container ?v1 - action_type)
	(turnon ?v0 - appliance ?v1 - action_type)
	(wait ?v0 - appliance ?v1 - action_type)
	(errormultipleliquidsmixed ?v0 - meal)
	(erroraddingredientstooearly ?v0 - liquid ?v1 - ingredient ?v2 - meal)
	(checkcereal ?v0 - meal ?v1 - action_type)
	(checkplainoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checkfruityoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checknuttyoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checkchocolateoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checkfruitychocolateoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checkpeanutbutterbananaoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checkpeanutbutterchocolateoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checknuttyfruityoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checknuttychocolateoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checknuttyfruitychocolateoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checknuttypeanutbutterbananaoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(checknuttypeanutbutterchocolateoatmeal ?v0 - container ?v1 - meal ?v2 - action_type)
	(putin ?v0 - container ?v1 - appliance ?v2 - action_type)
	(putinsink ?v0 - container ?v1 - appliance ?v2 - action_type)
	(putinmicrowave ?v0 - pastry ?v1 - action_type)
	(microwavepastry ?v0 - pastry ?v1 - action_type)
	(takeoutmicrowave ?v0 - pastry ?v1 - action_type)
	(checkpie ?v0 - meal ?v1 - action_type)
	(checkmuffin ?v0 - meal ?v1 - action_type)
	(checkroll ?v0 - meal ?v1 - action_type)
	(checkjellypastry ?v0 - meal ?v1 - action_type)
	(checkegg ?v0 - meal ?v1 - action_type)
	(collectwater ?v0 - container ?v1 - liquid ?v2 - action_type)
	(boilliquid ?v0 - meal ?v1 - container ?v2 - action_type)
	(reduceheat ?v0 - container ?v1 - meal ?v2 - action_type)
	(cookoatmeal ?v0 - appliance ?v1 - meal ?v2 - action_type)
	(serveoatmeal ?v0 - container ?v1 - container ?v2 - meal ?v3 - action_type)
  ) ; (:actions checknuttyfruityoatmeal reduceheat pourwater checkjellypastry checknuttyfruitychocolateoatmeal errormultipleliquidsmixed checkpie checkplainoatmeal checkcereal microwavepastry checknuttypeanutbutterbananaoatmeal checknuttyoatmeal checkpeanutbutterbananaoatmeal boilliquid checkfruitychocolateoatmeal takeoutmicrowave checkroll pour turnon checkchocolateoatmeal mix collectwater checknuttypeanutbutterchocolateoatmeal checknuttychocolateoatmeal checkmuffin wait checkegg checkpeanutbutterchocolateoatmeal serveoatmeal gather cookoatmeal checkfruityoatmeal putinmicrowave)

  
        (:action banana_in_storage
          :parameters (?x - object)
          :precondition (and (isbanana ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action blueberry_in_storage
          :parameters (?x - object)
          :precondition (and (isblueberry ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action chocolatechips_in_storage
          :parameters (?x - object)
          :precondition (and (ischocolatechips ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action cocopuffs_in_storage
          :parameters (?x - object)
          :precondition (and (iscocopuffs ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action egg_in_storage
          :parameters (?x - object)
          :precondition (and (isegg ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action jellypastry_in_storage
          :parameters (?x - object)
          :precondition (and (isjellypastry ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action milk_in_storage
          :parameters (?x - object)
          :precondition (and (ismilk ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action muffin_in_storage
          :parameters (?x - object)
          :precondition (and (ismuffin ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action oats_in_storage
          :parameters (?x - object)
          :precondition (and (isoats ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action peanutbutter_in_storage
          :parameters (?x - object)
          :precondition (and (ispeanutbutter ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action pie_in_storage
          :parameters (?x - object)
          :precondition (and (ispie ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action roll_in_storage
          :parameters (?x - object)
          :precondition (and (isroll ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action salt_in_storage
          :parameters (?x - object)
          :precondition (and (issalt ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action strawberry_in_storage
          :parameters (?x - object)
          :precondition (and (isstrawberry ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action water_in_storage
          :parameters (?x - object)
          :precondition (and (iswater ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action eatingspoon_in_storage
          :parameters (?x - object)
          :precondition (and (iseatingspoon ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action mixingspoon_in_storage
          :parameters (?x - object)
          :precondition (and (ismixingspoon ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action bowl_in_storage
          :parameters (?x - object)
          :precondition (and (isbowl ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action measuringcup_in_storage
          :parameters (?x - object)
          :precondition (and (ismeasuringcup ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action pan_in_storage
          :parameters (?x - object)
          :precondition (and (ispan ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action milk_in_storage
          :parameters (?x - object)
          :precondition (and (ismilk ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action water_in_storage
          :parameters (?x - object)
          :precondition (and (iswater ?x) (instorage ?x))
          :effect (dummyeffect ?x))

        
        (:action banana_in_workspace
          :parameters (?x - object)
          :precondition (and (isbanana ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action blueberry_in_workspace
          :parameters (?x - object)
          :precondition (and (isblueberry ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action chocolatechips_in_workspace
          :parameters (?x - object)
          :precondition (and (ischocolatechips ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action cocopuffs_in_workspace
          :parameters (?x - object)
          :precondition (and (iscocopuffs ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action egg_in_workspace
          :parameters (?x - object)
          :precondition (and (isegg ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action jellypastry_in_workspace
          :parameters (?x - object)
          :precondition (and (isjellypastry ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action milk_in_workspace
          :parameters (?x - object)
          :precondition (and (ismilk ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action muffin_in_workspace
          :parameters (?x - object)
          :precondition (and (ismuffin ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action oats_in_workspace
          :parameters (?x - object)
          :precondition (and (isoats ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action peanutbutter_in_workspace
          :parameters (?x - object)
          :precondition (and (ispeanutbutter ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action pie_in_workspace
          :parameters (?x - object)
          :precondition (and (ispie ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action roll_in_workspace
          :parameters (?x - object)
          :precondition (and (isroll ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action salt_in_workspace
          :parameters (?x - object)
          :precondition (and (issalt ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action strawberry_in_workspace
          :parameters (?x - object)
          :precondition (and (isstrawberry ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action water_in_workspace
          :parameters (?x - object)
          :precondition (and (iswater ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action eatingspoon_in_workspace
          :parameters (?x - object)
          :precondition (and (iseatingspoon ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action mixingspoon_in_workspace
          :parameters (?x - object)
          :precondition (and (ismixingspoon ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action bowl_in_workspace
          :parameters (?x - object)
          :precondition (and (isbowl ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action measuringcup_in_workspace
          :parameters (?x - object)
          :precondition (and (ismeasuringcup ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action pan_in_workspace
          :parameters (?x - object)
          :precondition (and (ispan ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action milk_in_workspace
          :parameters (?x - object)
          :precondition (and (ismilk ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action water_in_workspace
          :parameters (?x - object)
          :precondition (and (iswater ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))

        
        (:action bowl_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (isbowl ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action measuringcup_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (ismeasuringcup ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action pan_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (ispan ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action bowl_in_sink
          :parameters (?x - object ?a - appliance)
          :precondition (and (isbowl ?x) (issink ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action measuringcup_in_sink
          :parameters (?x - object ?a - appliance)
          :precondition (and (ismeasuringcup ?x) (issink ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action pan_in_sink
          :parameters (?x - object ?a - appliance)
          :precondition (and (ispan ?x) (issink ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action bowl_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (isbowl ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action measuringcup_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (ismeasuringcup ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action pan_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (ispan ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action egg_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (isegg ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action jellypastry_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (isjellypastry ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action muffin_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (ismuffin ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action pie_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (ispie ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action roll_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (isroll ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action bowl_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (isbowl ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action measuringcup_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (ismeasuringcup ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action pan_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (ispan ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))

        
        (:action microwave_is_on
          :parameters (?x - appliance)
          :precondition (and (ismicrowave ?x)(ison ?x))
          :effect (dummyeffect ?x))

        
        (:action sink_is_on
          :parameters (?x - appliance)
          :precondition (and (issink ?x)(ison ?x))
          :effect (dummyeffect ?x))

        
        (:action stove_is_on
          :parameters (?x - appliance)
          :precondition (and (isstove ?x)(ison ?x))
          :effect (dummyeffect ?x))

        
        (:action oatmeal_done
          :parameters (?x - food)
          :precondition (and (iscompletedoatmeal ?x))
          :effect (dummyeffect ?x))

        
        (:action pastry_done
          :parameters (?x - food)
          :precondition (and (iscompletedpastry ?x))
          :effect (dummyeffect ?x))

        
        (:action cereal_done
          :parameters (?x - food)
          :precondition (and (iscompletedcereal ?x))
          :effect (dummyeffect ?x))

        
        (:action banana_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isbanana ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action banana_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isbanana ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action banana_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isbanana ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action blueberry_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isblueberry ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action blueberry_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isblueberry ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action blueberry_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isblueberry ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action chocolatechips_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (ischocolatechips ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action chocolatechips_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (ischocolatechips ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action chocolatechips_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (ischocolatechips ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action cocopuffs_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (iscocopuffs ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action cocopuffs_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (iscocopuffs ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action cocopuffs_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (iscocopuffs ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action egg_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isegg ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action egg_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isegg ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action egg_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isegg ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action jellypastry_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isjellypastry ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action jellypastry_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isjellypastry ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action jellypastry_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isjellypastry ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action milk_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (ismilk ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action milk_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (ismilk ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action milk_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (ismilk ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action muffin_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (ismuffin ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action muffin_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (ismuffin ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action muffin_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (ismuffin ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action oats_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isoats ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action oats_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isoats ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action oats_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isoats ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action peanutbutter_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (ispeanutbutter ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action peanutbutter_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (ispeanutbutter ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action peanutbutter_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (ispeanutbutter ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action pie_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (ispie ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action pie_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (ispie ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action pie_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (ispie ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action roll_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isroll ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action roll_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isroll ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action roll_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isroll ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action salt_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (issalt ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action salt_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (issalt ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action salt_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (issalt ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action strawberry_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isstrawberry ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action strawberry_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isstrawberry ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action strawberry_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isstrawberry ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action water_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (iswater ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action water_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (iswater ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action water_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (iswater ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action milk_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (ismilk ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action milk_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (ismilk ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action milk_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (ismilk ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action water_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (iswater ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action water_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (iswater ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action water_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (iswater ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action cereal_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (iscereal ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action cereal_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (iscereal ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action cereal_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (iscereal ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action oatmeal_in_bowl
          :parameters (?x - object ?c - container)
          :precondition (and (isoatmeal ?x) (isbowl ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action oatmeal_in_measuringcup
          :parameters (?x - object ?c - container)
          :precondition (and (isoatmeal ?x) (ismeasuringcup ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action oatmeal_in_pan
          :parameters (?x - object ?c - container)
          :precondition (and (isoatmeal ?x) (ispan ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))

        
        (:action is_simmering
          :parameters (?x - food)
          :precondition (and (simmering ?x))
          :effect (dummyeffect ?x))

        
        (:action is_boiling
          :parameters (?x - food)
          :precondition (and (boiling ?x))
          :effect (dummyeffect ?x))

        
        (:action is_heated
          :parameters (?x - food)
          :precondition (and (heated ?x))
          :effect (dummyeffect ?x))

        
        (:action is_mixed
          :parameters (?x - food)
          :precondition (and (mixed ?x))
          :effect (dummyeffect ?x))

        
        (:action is_warmed
          :parameters (?x - food)
          :precondition (and (warmed ?x))
          :effect (dummyeffect ?x))

        
        (:action holding_mixingspoon
          :parameters (?x - tool)
          :precondition (and (ismixingspoon ?x)(inhand  ?x))
          :effect (dummyeffect ?x))

        
        (:action error_milk_and_water_in_pan
          :parameters (?m - liquid ?w - liquid ?p - container)
          :precondition (and (ispan ?p)
                        (iswater ?w) (incontainer ?w ?p)
                        (ismilk ?m) (incontainer ?m ?p)
        )
          :effect (dummyeffect ?m))

        
)
        