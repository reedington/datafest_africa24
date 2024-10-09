-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODhost_summary_by_file_ioE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `mydb` DEFAULT CHARACTER SET utf8 ;
USE `mydb` ;

-- -----------------------------------------------------
-- Table `mydb`.`student_information`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `mydb`.`student_information` ;

CREATE TABLE IF NOT EXISTS `mydb`.`student_information` (
  `StudentID` INT NOT NULL,
  `Age` TINYINT(1) NULL,
  `Gender` ENUM('0', '1') NULL,
  `StateOfOrigin` VARCHAR(45) NULL,
  `EconomicBackground` ENUM('0', '1', '2') NULL,
  `AttendanceRate` FLOAT NULL,
  `Class` ENUM('SS1', 'SS2', 'SS3') NULL,
  `OverallPercentage` FLOAT NULL,
  PRIMARY KEY (`StudentID`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`performance_prediction_labels`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `mydb`.`performance_prediction_labels` ;

CREATE TABLE IF NOT EXISTS `mydb`.`performance_prediction_labels` (
  `StudentID` INT NOT NULL,
  `AcademicYear` VARCHAR(45) NULL,
  `RiskOfFailure` TINYINT NULL,
  `ImprovementFlag` TINYINT NULL,
  `StudentID1` INT NOT NULL,
  PRIMARY KEY (`StudentID1`),
  CONSTRAINT `StudentID1`
    FOREIGN KEY (`StudentID1`)
    REFERENCES `mydb`.`student_information` (`StudentID`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`extracurricular_activities`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `mydb`.`extracurricular_activities` ;

CREATE TABLE IF NOT EXISTS `mydb`.`extracurricular_activities` (
  `StudentID` INT NOT NULL,
  `ActivityType` VARCHAR(45) NULL,
  `ParticipationLevel` ENUM('None', 'Low', 'Medium', 'High') NULL,
  `StudentID1` INT NOT NULL,
  PRIMARY KEY (`StudentID1`),
  CONSTRAINT `StudentID3`
    FOREIGN KEY (`StudentID1`)
    REFERENCES `mydb`.`student_information` (`StudentID`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`behavioral_and_socioeconomic_factors`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `mydb`.`behavioral_and_socioeconomic_factors` ;

CREATE TABLE IF NOT EXISTS `mydb`.`behavioral_and_socioeconomic_factors` (
  `StudentID` INT NOT NULL,
  `StudyTimeWeekly` FLOAT NULL,
  `ParentalSupportLevel` ENUM('Low', 'Medium', 'High') NULL,
  `BehaviouralScore` ENUM('1', '2', '3', '4') NULL,
  `TeacherStudentRatio` FLOAT NULL,
  `Tutoring` ENUM('0', '1') NULL,
  `Absences` INT NULL,
  `ParentalEducation` ENUM('0', '1', '2', '3', '4') NULL,
  `StudentID1` INT NOT NULL,
  PRIMARY KEY (`StudentID1`),
  CONSTRAINT `StudentID`
    FOREIGN KEY (`StudentID1`)
    REFERENCES `mydb`.`student_information` (`StudentID`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`teacher_information`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `mydb`.`teacher_information` ;

CREATE TABLE IF NOT EXISTS `mydb`.`teacher_information` (
  `TeacherID` INT NOT NULL,
  `Subject` VARCHAR(45) NULL,
  `YearsOfExperience` TINYINT(1) NULL,
  `QualificationLevel` ENUM('B.Ed', 'NCE', 'M.Ed') NULL,
  PRIMARY KEY (`TeacherID`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `mydb`.`academic_performance`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `mydb`.`academic_performance` ;

CREATE TABLE IF NOT EXISTS `mydb`.`academic_performance` (
  `StudentID` INT NOT NULL,
  `AcademicYear` VARCHAR(45) NULL,
  `Term` ENUM('First Term', 'Second Term', 'Third Term') NULL,
  `Subject` VARCHAR(45) NULL,
  `Score` FLOAT NULL,
  `Class` ENUM('SS1', 'SS2', 'SS3') NULL,
  `StudentID1` INT NOT NULL,
  `TeacherID` INT NOT NULL,
  PRIMARY KEY (`StudentID1`, `TeacherID`),
  CONSTRAINT `StudentID2`
    FOREIGN KEY (`StudentID1`)
    REFERENCES `mydb`.`student_information` (`StudentID`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `TeacherID1`
    FOREIGN KEY (`TeacherID`)
    REFERENCES `mydb`.`teacher_information` (`TeacherID`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;

CREATE INDEX `TeacherID1_idx` ON `mydb`.`academic_performance` (`TeacherID` ASC) VISIBLE;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
